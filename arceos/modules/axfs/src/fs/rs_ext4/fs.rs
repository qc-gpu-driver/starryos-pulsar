#![cfg(feature = "rsext4")]

use alloc::{sync::Arc};
use core::cell::OnceCell;
use axdriver::prelude::BlockDriverOps;
use alloc::vec;
use axdriver::AxBlockDevice;
use axfs_ng_vfs::{
    path::MAX_NAME_LEN, DirEntry, DirNode, Filesystem, FilesystemOps, Reference, StatFs,
    VfsResult,
};
use kspin::{SpinNoPreempt as Mutex, SpinNoPreemptGuard as MutexGuard};

use rsext4::ext4_backend::ext4::Ext4FileSystem;
use rsext4::{BlockDevice as RsBlockDevice, BlockDevError, BlockDevResult, Jbd2Dev};

use super::{inode::Rsext4Node, util::{into_vfs_err, into_vfs_fs_err}};

const EXT4_ROOT_INO: u32 = 2;

pub struct Rsext4Disk(pub AxBlockDevice);

impl RsBlockDevice for Rsext4Disk {
    fn write(&mut self, buffer: &[u8], block_id: u32, count: u32) -> BlockDevResult<()> {
        let dev_bs = self.0.block_size();
        let fs_bs = rsext4::BLOCK_SIZE;
        if fs_bs % dev_bs != 0 {
            return Err(BlockDevError::InvalidBlockSize {
                size: dev_bs,
                expected: fs_bs,
            });
        }
        let ratio = (fs_bs / dev_bs) as u64;
        let total_bytes = (count as usize)
            .saturating_mul(fs_bs)
            .min(buffer.len());

        for (i, fs_block) in buffer[..total_bytes].chunks(fs_bs).enumerate() {
            for (j, dev_chunk) in fs_block.chunks(dev_bs).enumerate() {
                let mut blk = vec![0u8; dev_bs];
                blk[..dev_chunk.len()].copy_from_slice(dev_chunk);
                let dev_block_id = (block_id as u64 + i as u64) * ratio + j as u64;
                self.0
                    .write_block(dev_block_id, &blk)
                    .map_err(|_| BlockDevError::WriteError)?;
            }
        }
        Ok(())
    }

    fn read(&mut self, buffer: &mut [u8], block_id: u32, count: u32) -> BlockDevResult<()> {
        let dev_bs = self.0.block_size();
        let fs_bs = rsext4::BLOCK_SIZE;
        if fs_bs % dev_bs != 0 {
            return Err(BlockDevError::InvalidBlockSize {
                size: dev_bs,
                expected: fs_bs,
            });
        }
        let ratio = (fs_bs / dev_bs) as u64;
        let total_bytes = (count as usize)
            .saturating_mul(fs_bs)
            .min(buffer.len());

        for (i, fs_block) in buffer[..total_bytes].chunks_mut(fs_bs).enumerate() {
            for (j, dev_chunk) in fs_block.chunks_mut(dev_bs).enumerate() {
                let mut blk = vec![0u8; dev_bs];
                let dev_block_id = (block_id as u64 + i as u64) * ratio + j as u64;
                self.0
                    .read_block(dev_block_id, &mut blk)
                    .map_err(|_| BlockDevError::ReadError)?;
                dev_chunk.copy_from_slice(&blk[..dev_chunk.len()]);
            }
        }
        Ok(())
    }

    fn open(&mut self) -> BlockDevResult<()> {
        Ok(())
    }

    fn close(&mut self) -> BlockDevResult<()> {
        Ok(())
    }

    fn total_blocks(&self) -> u64 {
        let dev_bs = self.0.block_size();
        let fs_bs = rsext4::BLOCK_SIZE;
        if fs_bs % dev_bs != 0 {
            return 0;
        }
        let ratio = (fs_bs / dev_bs) as u64;
        self.0.num_blocks() / ratio
    }

    fn block_size(&self) -> u32 {
        rsext4::BLOCK_SIZE as u32
    }

    fn flush(&mut self) -> BlockDevResult<()> {
        self.0.flush().map_err(|_| BlockDevError::WriteError)
    }
}

pub struct Rsext4FilesystemInner {
    pub fs: Ext4FileSystem,
    pub dev: Jbd2Dev<Rsext4Disk>,
}

pub struct Rsext4Filesystem {
    inner: Mutex<Rsext4FilesystemInner>,
    root_dir: OnceCell<DirEntry>,
}

impl Rsext4Filesystem {
    pub fn new(dev: AxBlockDevice) -> VfsResult<Filesystem> {
        let mut jbd = Jbd2Dev::initial_jbd2dev(0, Rsext4Disk(dev), true);
        let fs = Ext4FileSystem::mount(&mut jbd).map_err(into_vfs_fs_err)?;

        let fs = Arc::new(Self {
            inner: Mutex::new(Rsext4FilesystemInner { fs, dev: jbd }),
            root_dir: OnceCell::new(),
        });

        let _ = fs.root_dir.set(DirEntry::new_dir(
            |this| DirNode::new(Rsext4Node::new(fs.clone(), "/", EXT4_ROOT_INO, Some(this))),
            Reference::root(),
        ));

        Ok(Filesystem::new(fs))
    }

    pub(crate) fn lock<'a>(&'a self) -> MutexGuard<'a,Rsext4FilesystemInner> {
        self.inner.lock()
    }
}

unsafe impl Send for Rsext4Filesystem {}
unsafe impl Sync for Rsext4Filesystem {}

impl FilesystemOps for Rsext4Filesystem {
    fn name(&self) -> &str {
        "ext4"
    }

    fn root_dir(&self) -> DirEntry {
        self.root_dir.get().unwrap().clone()
    }

    fn stat(&self) -> VfsResult<StatFs> {
        let inner = self.lock();
        let sb = &inner.fs.superblock;
        Ok(StatFs {
            fs_type: 0xef53,
            block_size: rsext4::BLOCK_SIZE as _,
            blocks: sb.blocks_count() as _,
            blocks_free: sb.free_blocks_count() as _,
            blocks_available: sb.free_blocks_count() as _,
            file_count: sb.s_inodes_count as _,
            free_file_count: sb.s_free_inodes_count as _,
            name_length: MAX_NAME_LEN as _,
            fragment_size: 0,
            mount_flags: 0,
        })
    }

    fn flush(&self) -> VfsResult<()> {
        let fs = &mut self.lock().fs;
        let block_dev = &mut self.lock().dev;
                
        fs.bitmap_cache.flush_all(block_dev).map_err(into_vfs_err)?;
        fs.inodetable_cahce.flush_all(block_dev).map_err(into_vfs_err)?;
        fs.datablock_cache.flush_all(block_dev).map_err(into_vfs_err)?;

        // 4. Update superblock
        fs.sync_superblock(block_dev).map_err(into_vfs_err)?;

        // Write back group descriptors
        fs.sync_group_descriptors(block_dev).map_err(into_vfs_err)
    }
}

