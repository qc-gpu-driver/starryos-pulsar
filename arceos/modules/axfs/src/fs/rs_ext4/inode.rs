#![cfg(feature = "rsext4")]

use alloc::{borrow::ToOwned, string::String, sync::Arc, vec::Vec};
use core::{any::Any, task::Context, time::Duration};

use axfs_ng_vfs::{
    DirEntry, DirEntrySink, DirNode, DirNodeOps, FileNode, FileNodeOps, FilesystemOps, Metadata,
    MetadataUpdate, NodeFlags, NodeOps, NodePermission, NodeType, Reference, VfsError, VfsResult,
    WeakDirEntry, DeviceId,
};
use axpoll::{IoEvents, Pollable};

use rsext4::ext4_backend::entries::{Ext4DirEntry2, classic_dir};
use rsext4::ext4_backend::disknode::{Ext4Extent, Ext4Inode};

use super::fs::{Rsext4Disk, Rsext4Filesystem};
use super::util::{into_vfs_err, into_vfs_type};

fn read_symlink_target(
    fs: &mut rsext4::Ext4FileSystem,
    dev: &mut rsext4::Jbd2Dev<Rsext4Disk>,
    inode: &mut Ext4Inode,
) -> VfsResult<Vec<u8>> {
    let size = inode.size() as usize;
    if size == 0 {
        return Ok(Vec::new());
    }
    if size <= 60 {
        let mut raw = [0u8; 60];
        for (i, word) in inode.i_block.iter().take(15).enumerate() {
            raw[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        return Ok(raw[..size].to_vec());
    }

    let block_bytes = rsext4::BLOCK_SIZE;
    let total_blocks = size.div_ceil(block_bytes);
    let mut buf = Vec::with_capacity(size);
    if inode.have_extend_header_and_use_extend() {
        let blocks = rsext4::ext4_backend::loopfile::resolve_inode_block_allextend(fs, dev, inode)
            .map_err(into_vfs_err)?;
        for phys in blocks {
            let cached = fs
                .datablock_cache
                .get_or_load(dev, phys.1)
                .map_err(into_vfs_err)?;
            buf.extend_from_slice(&cached.data[..block_bytes]);
            if buf.len() >= size {
                break;
            }
        }
    } else {
        for lbn in 0..total_blocks {
            let phys = match rsext4::ext4_backend::loopfile::resolve_inode_block(dev, inode, lbn as u32)
                .map_err(into_vfs_err)?
            {
                Some(b) => b,
                None => break,
            };
            let cached = fs
                .datablock_cache
                .get_or_load(dev, phys as u64)
                .map_err(into_vfs_err)?;
            buf.extend_from_slice(&cached.data[..block_bytes]);
        }
    }
    buf.truncate(size);
    Ok(buf)
}

pub struct Rsext4Node {
    pub fs: Arc<Rsext4Filesystem>,
    pub path: String,
    pub ino: u32,
    pub this: Option<WeakDirEntry>,
}

impl Rsext4Node {
    pub(crate) fn new(
        fs: Arc<Rsext4Filesystem>,
        path: impl Into<String>,
        ino: u32,
        this: Option<WeakDirEntry>,
    ) -> Arc<Self> {
        Arc::new(Self {
            fs,
            path: path.into(),
            ino,
            this,
        })
    }

    fn child_path(&self, name: &str) -> String {
        if self.path == "/" {
            let mut p = String::from("/");
            p.push_str(name);
            p
        } else {
            let mut p = self.path.clone();
            p.push('/');
            p.push_str(name);
            p
        }
    }

    fn create_entry(&self, name: &str, child_path: String, child_ino: u32) -> VfsResult<DirEntry> {
        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);
        let inode = fs.get_inode_by_num(dev, child_ino).map_err(into_vfs_err)?;
        let node_type = into_vfs_type(&inode);

        let reference = Reference::new(
            self.this.as_ref().and_then(WeakDirEntry::upgrade),
            name.to_owned(),
        );

        Ok(if node_type == NodeType::Directory {
            DirEntry::new_dir(
                |this| DirNode::new(Rsext4Node::new(self.fs.clone(), child_path, child_ino, Some(this))),
                reference,
            )
        } else {
            DirEntry::new_file(
                FileNode::new(Rsext4Node::new(self.fs.clone(), child_path, child_ino, None)),
                node_type,
                reference,
            )
        })
    }

    fn lookup_locked(&self, name: &str) -> VfsResult<DirEntry> {
        let child_path = self.child_path(name);
        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);
        let res = rsext4::get_inode_with_num(fs, dev, &child_path)
            .map_err(into_vfs_err)?
            .ok_or(VfsError::NotFound)?;
        let _ = inner;
        self.create_entry(name, child_path, res.0)
    }
}

impl NodeOps for Rsext4Node {
    fn inode(&self) -> u64 {
        self.ino as _
    }

    fn metadata(&self) -> VfsResult<Metadata> {
        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);
        let inode = fs.get_inode_by_num(dev, self.ino).map_err(into_vfs_err)?;
        let node_type = into_vfs_type(&inode);

        Ok(Metadata {
            inode: self.ino as _,
            device: 0,
            nlink: inode.i_links_count as _,
            mode: NodePermission::from_bits_truncate((inode.i_mode & 0o777) as u16),
            node_type,
            uid: inode.i_uid as _,
            gid: inode.i_gid as _,
            size: inode.size(),
            block_size: rsext4::BLOCK_SIZE as _,
            blocks: ((inode.blocks_count() as u64) * 512) / (rsext4::BLOCK_SIZE as u64),
            rdev: DeviceId::default(),
            atime: Duration::from_secs(inode.i_atime as _),
            mtime: Duration::from_secs(inode.i_mtime as _),
            ctime: Duration::from_secs(inode.i_ctime as _),
        })
    }

    fn update_metadata(&self, _update: MetadataUpdate) -> VfsResult<()> {
        // rsext4-ext4 backend does not fully support updating inode timestamps/mode/owner yet.
        // Return success to avoid breaking common userspace tools (e.g. apk) that treat
        // utimensat/chmod/chown failures as fatal.
        Ok(())
    }

    fn len(&self) -> VfsResult<u64> {
        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);
        let inode = fs.get_inode_by_num(dev, self.ino).map_err(into_vfs_err)?;
        Ok(inode.size())
    }

    fn filesystem(&self) -> &dyn FilesystemOps {
        &*self.fs
    }

    fn sync(&self, _data_only: bool) -> VfsResult<()> {
        self.fs.flush()
    }

    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn flags(&self) -> NodeFlags {
        NodeFlags::BLOCKING | NodeFlags::NON_CACHEABLE
    }
}

impl FileNodeOps for Rsext4Node {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> VfsResult<usize> {

        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);
        let mut inode = fs.get_inode_by_num(dev, self.ino).map_err(into_vfs_err)?;
        let file_size = inode.size() as usize;

        let off = offset as usize;
        if off >= file_size {
            return Ok(0);
        }
        let n = core::cmp::min(buf.len(), file_size - off);

        if inode.is_symlink() {
            let data = read_symlink_target(fs, dev, &mut inode)?;
            let n2 = core::cmp::min(n, data.len().saturating_sub(off));
            if n2 == 0 {
                return Ok(0);
            }
            buf[..n2].copy_from_slice(&data[off..off + n2]);

          
            return Ok(n2);
        }

        let block_bytes = rsext4::BLOCK_SIZE;
        let start_lbn = off / block_bytes;
        let end_lbn = (off + n - 1) / block_bytes;

        let mut copied = 0usize;
        let mut inode_for_map = inode.clone();
        for lbn in start_lbn..=end_lbn {
            let phys = match rsext4::ext4_backend::loopfile::resolve_inode_block(
                dev,
                &mut inode_for_map,
                lbn as u32,
            )
            .map_err(into_vfs_err)?
            {
                Some(b) => b as u64,
                None => {
                    // sparse/hole block: fill zeros
                    let dst_off = lbn * block_bytes;
                    let dst_start = off.max(dst_off) - off;
                    let dst_end = (off + n).min(dst_off + block_bytes) - off;
                    buf[dst_start..dst_end].fill(0);
                    continue;
                }
            };

            let cached = fs
                .datablock_cache
                .get_or_load(dev, phys)
                .map_err(into_vfs_err)?;

            let blk = &cached.data[..block_bytes];
            let src_off = if lbn == start_lbn { off % block_bytes } else { 0 };
            let to_copy = core::cmp::min(block_bytes - src_off, n - copied);
            buf[copied..copied + to_copy].copy_from_slice(&blk[src_off..src_off + to_copy]);
            copied += to_copy;
            if copied >= n {
                break;
            }
        }

      
        Ok(copied)
    }

    fn write_at(&self, buf: &[u8], offset: u64) -> VfsResult<usize> {
       
        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);

        rsext4::write_file_with_ino(dev, fs, self.ino, offset, buf)
            .map_err(into_vfs_err)?;

        Ok(buf.len())
    }

    fn append(&self, buf: &[u8]) -> VfsResult<(usize, u64)> {
        
        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);
        let inode = fs.get_inode_by_num(dev, self.ino).map_err(into_vfs_err)?;
        let offset = inode.size();
        rsext4::write_file_with_ino(dev, fs, self.ino, offset, buf)
            .map_err(into_vfs_err)?;

      
        Ok((buf.len(), offset + buf.len() as u64))
    }

    fn set_len(&self, len: u64) -> VfsResult<()> {
        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);
        rsext4::truncate_with_ino(dev, fs, self.ino, len).map_err(into_vfs_err)
    }

    fn set_symlink(&self, target: &str) -> VfsResult<()> {
        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);

        let mut inode = fs.get_inode_by_num(dev, self.ino).map_err(into_vfs_err)?;
        // If the inode currently has blocks (e.g. created as regular file), free them first.
        // This avoids leaks and keeps metadata consistent.
        if inode.blocks_count() != 0 {
            if inode.have_extend_header_and_use_extend() {
                let used_blocks = rsext4::ext4_backend::loopfile::resolve_inode_block_allextend(
                    fs,
                    dev,
                    &mut inode,
                )
                .map_err(into_vfs_err)?;
                for blk in used_blocks {
                    fs.free_block(dev, blk.1).map_err(into_vfs_err)?;
                }
            } else {
                // Only direct pointers are supported by this backend.
                for w in inode.i_block.iter_mut().take(12) {
                    let phys = *w as u64;
                    if phys != 0 {
                        fs.free_block(dev, phys).map_err(into_vfs_err)?;
                    }
                    *w = 0;
                }
            }
        }

        let target_bytes = target.as_bytes();
        let target_len = target_bytes.len() as u64;

        // Convert inode to symlink
        inode.i_mode = Ext4Inode::S_IFLNK | 0o777;
        inode.i_flags &= !Ext4Inode::EXT4_EXTENTS_FL;//symbol链接不允许存在extent标志，但是可以使用extendheader
        inode.i_size_lo = (target_len & 0xffff_ffff) as u32;
        inode.i_size_high = (target_len >> 32) as u32;
        inode.i_blocks_lo = 0;
        inode.l_i_blocks_high = 0;
        inode.i_block = [0; 15];

        if target_len == 0 {
            // empty
        } else if target_len as usize <= 60 {
            // fast symlink stored inline in i_block
            let mut raw = [0u8; 60];
            raw[..target_len as usize].copy_from_slice(target_bytes);
            for (i, word) in inode.i_block.iter_mut().take(15).enumerate() {
                *word = u32::from_le_bytes([
                    raw[i * 4],
                    raw[i * 4 + 1],
                    raw[i * 4 + 2],
                    raw[i * 4 + 3],
                ]);
            }
        } else {
            // slow symlink stored in data blocks
            // Prefer extent mode when available.
            if fs.superblock.has_extents() {
                inode.i_flags |= Ext4Inode::EXT4_EXTENTS_FL;
                inode.write_extend_header();

                let block_bytes = rsext4::BLOCK_SIZE as u64;
                let total_blocks = target_len.div_ceil(block_bytes);
                let mut new_blocks_map: Vec<(u32, u64)> = Vec::new();

                for lbn in 0..(total_blocks as u32) {
                    let phys = fs.alloc_block(dev).map_err(into_vfs_err)?;
                    let start = (lbn as usize) * rsext4::BLOCK_SIZE;
                    let end = core::cmp::min(start + rsext4::BLOCK_SIZE, target_bytes.len());
                    fs.datablock_cache.modify_new(phys, |data| {
                        data.fill(0);
                        data[..(end - start)].copy_from_slice(&target_bytes[start..end]);
                    });
                    new_blocks_map.push((lbn, phys));
                }

                let mut tree = rsext4::ext4_backend::extents_tree::ExtentTree::new(&mut inode);
                let mut idx = 0usize;
                while idx < new_blocks_map.len() {
                    let (start_lbn, start_phys) = new_blocks_map[idx];
                    let mut run_len: u32 = 1;
                    let mut last_lbn = start_lbn;
                    let mut last_phys = start_phys;
                    idx += 1;
                    while idx < new_blocks_map.len() {
                        let (cur_lbn, cur_phys) = new_blocks_map[idx];
                        if cur_lbn == last_lbn + 1 && cur_phys == last_phys + 1 {
                            run_len = run_len.saturating_add(1);
                            last_lbn = cur_lbn;
                            last_phys = cur_phys;
                            idx += 1;
                        } else {
                            break;
                        }
                    }
                    let ext = Ext4Extent::new(
                        start_lbn,
                        start_phys,
                        run_len as u16,
                    );
                    tree.insert_extent(fs, ext, dev).map_err(into_vfs_err)?;
                }

                // ExtentTree insertion may allocate extra metadata blocks (index/leaf nodes) and
                // already accounts them into inode.i_blocks*. Do not overwrite it.
                let cur = inode.blocks_count();
                let data_sectors = total_blocks.saturating_mul((rsext4::BLOCK_SIZE / 512) as u64);
                let newv = cur.saturating_add(data_sectors);
                inode.i_blocks_lo = (newv & 0xffff_ffff) as u32;
                inode.l_i_blocks_high = ((newv >> 32) & 0xffff) as u16;
            } else {
                // Fallback: direct blocks only (max 12)
                let block_bytes = rsext4::BLOCK_SIZE as u64;
                let total_blocks = target_len.div_ceil(block_bytes);
                if total_blocks > 12 {
                    return Err(VfsError::Unsupported);
                }
                for lbn in 0..(total_blocks as u32) {
                    let phys = fs.alloc_block(dev).map_err(into_vfs_err)?;
                    let start = (lbn as usize) * rsext4::BLOCK_SIZE;
                    let end = core::cmp::min(start + rsext4::BLOCK_SIZE, target_bytes.len());
                    fs.datablock_cache.modify_new(phys, |data| {
                        data.fill(0);
                        data[..(end - start)].copy_from_slice(&target_bytes[start..end]);
                    });
                    inode.i_block[lbn as usize] = phys as u32;
                }
                let iblocks_used = total_blocks.saturating_mul((rsext4::BLOCK_SIZE / 512) as u64);
                inode.i_blocks_lo = (iblocks_used & 0xffff_ffff) as u32;
                inode.l_i_blocks_high = ((iblocks_used >> 32) & 0xffff) as u16;
            }
        }

        fs.modify_inode(dev, self.ino, |td| {
            *td = inode;
        })
        .map_err(into_vfs_err)?;

        Ok(())
    }
}

impl Pollable for Rsext4Node {
    fn poll(&self) -> IoEvents {
        IoEvents::IN | IoEvents::OUT
    }

    fn register(&self, _context: &mut Context<'_>, _events: IoEvents) {}
}

impl DirNodeOps for Rsext4Node {
    fn read_dir(&self, offset: u64, sink: &mut dyn DirEntrySink) -> VfsResult<usize> {
       
        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);
        let dir_inode = fs
            .get_inode_by_num(dev, self.ino)
            .map_err(into_vfs_err)?;
        if !dir_inode.is_dir() {
            return Err(VfsError::NotADirectory);
        }

        let total_size = dir_inode.size() as usize;
        let block_bytes = rsext4::BLOCK_SIZE;
        let total_blocks = if total_size == 0 {
            0
        } else {
            total_size.div_ceil(block_bytes)
        };

        let mut entries: alloc::vec::Vec<(String, u32)> = alloc::vec::Vec::new();
        fn scan_block(
            fs: &mut rsext4::Ext4FileSystem,
            dev: &mut rsext4::Jbd2Dev<Rsext4Disk>,
            block_bytes: usize,
            phys_u64: u64,
            entries: &mut alloc::vec::Vec<(String, u32)>,
        ) -> VfsResult<()> {
            let cached = fs.datablock_cache.get_or_load(dev, phys_u64).map_err(into_vfs_err)?;
            let block_data = &cached.data[..block_bytes];
            for info in classic_dir::list_entries(block_data) {
                let name = core::str::from_utf8(info.name)
                    .map_err(|_| VfsError::InvalidData)?
                    .to_owned();
                if name == "." || name == ".." {
                    continue;
                }
                entries.push((name, info.inode as u32));
            }
            Ok(())
        }

        if dir_inode.have_extend_header_and_use_extend() {
            let blocks = rsext4::ext4_backend::loopfile::resolve_inode_block_allextend(
                fs,
                dev,
                &mut dir_inode.clone(),
            )
            .map_err(into_vfs_err)?;
            for phys in blocks {
                scan_block(fs, dev, block_bytes, phys.1, &mut entries)?;
            }
        } else {
            for lbn in 0..total_blocks {
                let phys = match rsext4::ext4_backend::loopfile::resolve_inode_block(
                    
                    dev,
                    &mut dir_inode.clone(),
                    lbn as u32,
                )
                .map_err(into_vfs_err)? {
                    Some(b) => b,
                    None => continue,
                };
                scan_block(fs, dev, block_bytes, phys as u64, &mut entries)?;
            }
        }

        let start = offset as usize;
        if start >= entries.len() {
            return Ok(0);
        }

        let mut count = 0;
        for (idx, (name, ino)) in entries.into_iter().enumerate().skip(start) {
            let inode = fs.get_inode_by_num(dev, ino).map_err(into_vfs_err)?;
            let node_type = into_vfs_type(&inode);
            let next_off = (idx + 1) as u64;
            if !sink.accept(&name, ino as u64, node_type, next_off) {
                break;
            }
            count += 1;
        }

        
        Ok(count)
    }

    fn lookup(&self, name: &str) -> VfsResult<DirEntry> {
        self.lookup_locked(name)
    }

    fn create(
        &self,
        name: &str,
        node_type: NodeType,
        _permission: NodePermission,
    ) -> VfsResult<DirEntry> {
        let child_path = self.child_path(name);
        let mut inner = self.fs.lock();

        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);

        if rsext4::get_inode_with_num(fs, dev, &child_path)
            .map_err(into_vfs_err)?
            .is_some()
        {
            return Err(VfsError::AlreadyExists);
        }

        match node_type {
            NodeType::Directory => {
                let Some((ino, _inode)) = rsext4::mkdir_with_ino(dev, fs, &child_path) else {
                    error!("rsext4 create mkdir failed path={}", child_path);
                    return Err(VfsError::Io);
                };
                let _ = inner;
                return self.create_entry(name, child_path, ino);
            }
            NodeType::RegularFile => {
                let Some((ino, _inode)) = rsext4::mkfile_with_ino(dev, fs, &child_path, None,None) else {
                    error!("rsext4 create mkfile failed path={}", child_path);
                    return Err(VfsError::Io);
                };
                let _ = inner;
                return self.create_entry(name, child_path, ino);
            }
            NodeType::Symlink => {
                // Create an empty inode first, then FileNodeOps::set_symlink will convert it
                // into a symlink and store the target.
                let Some((ino, _inode)) = rsext4::mkfile_with_ino(dev, fs, &child_path, None,Some(Ext4DirEntry2::EXT4_FT_SYMLINK)) else {
                    error!("rsext4 create symlink(mkfile) failed path={}", child_path);
                    return Err(VfsError::Io);
                };
                let _ = inner;
                return self.create_entry(name, child_path, ino);
            }
            _ => return Err(VfsError::Unsupported),
        }


    }

    fn link(&self, _name: &str, _node: &DirEntry) -> VfsResult<DirEntry> {
        let target = _node.downcast::<Self>()?;
        if !Arc::ptr_eq(&self.fs, &target.fs) {
            return Err(VfsError::InvalidInput);
        }

        let link_path = self.child_path(_name);
        let linked_path = target.path.clone();

        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);

        if rsext4::get_inode_with_num(fs, dev, &link_path)
            .map_err(into_vfs_err)?
            .is_some()
        {
            return Err(VfsError::AlreadyExists);
        }

        // rsext4 hardlink: create a new dir entry at link_path pointing to linked_path's inode.
        rsext4::link(fs, dev, &link_path, &linked_path);

        let _ = inner;
        self.lookup_locked(_name)
    }

    fn unlink(&self, _name: &str) -> VfsResult<()> {
        let child_path = self.child_path(_name);
        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);

        if rsext4::get_inode_with_num(fs, dev, &child_path)
            .map_err(into_vfs_err)?
            .is_none()
        {
            return Err(VfsError::NotFound);
        }

        rsext4::unlink(fs, dev, &child_path);

        if rsext4::get_inode_with_num(fs, dev, &child_path)
            .map_err(into_vfs_err)?
            .is_some()
        {
            return Err(VfsError::Io);
        }

        Ok(())
    }

    fn rename(&self, _src_name: &str, _dst_dir: &DirNode, _dst_name: &str) -> VfsResult<()> {
        let dst_node = _dst_dir.downcast::<Self>()?;
        if !Arc::ptr_eq(&self.fs, &dst_node.fs) {
            return Err(VfsError::InvalidInput);
        }

        let src_path = self.child_path(_src_name);
        let dst_path = dst_node.child_path(_dst_name);

        let mut inner = self.fs.lock();
        let inner = &mut *inner;
        let (fs, dev) = (&mut inner.fs, &mut inner.dev);

        if let Err(e) = rsext4::rename(dev, fs, &src_path, &dst_path).map_err(into_vfs_err) {
            error!(
                "rsext4 rename failed: src={} dst={} err={:?}",
                src_path,
                dst_path,
                e
            );
            return Err(e);
        }
        Ok(())
    }
}
