
use axfs_ng_vfs::{NodeType, VfsError};

use rsext4::ext4_backend::disknode::Ext4Inode;
use rsext4::BlockDevError;
use rsext4::RSEXT4Error;

pub fn into_vfs_err(err: BlockDevError) -> VfsError {
    use BlockDevError::*;
    match err {
        InvalidInput | BufferTooSmall { .. } | AlignmentError { .. } => VfsError::InvalidInput,
        BlockOutOfRange { .. } | InvalidBlockSize { .. } => VfsError::InvalidData,
        DeviceNotOpen | DeviceClosed | DeviceBusy | Timeout => VfsError::Io,
        ReadOnly => VfsError::PermissionDenied,
        NoSpace => VfsError::StorageFull,
        Unsupported => VfsError::Unsupported,
        PermissionDenied => VfsError::PermissionDenied,
        Corrupted | ChecksumError => VfsError::InvalidData,
        ReadError | WriteError | IoError | Unknown => VfsError::Io,
    }
}

pub fn into_vfs_fs_err(err: RSEXT4Error) -> VfsError {
    use RSEXT4Error::*;
    match err {
        IoError => VfsError::Io,
        InvalidMagic | InvalidSuperblock => VfsError::InvalidData,
        FilesystemHasErrors => VfsError::InvalidData,
        UnsupportedFeature => VfsError::Unsupported,
        AlreadyMounted => VfsError::AlreadyExists,
    }
}

pub fn into_vfs_type(inode: &Ext4Inode) -> NodeType {
    if inode.is_dir() {
        NodeType::Directory
    } else if inode.is_symlink() {
        NodeType::Symlink
    } else if inode.is_file() {
        NodeType::RegularFile
    } else {
        NodeType::Unknown
    }
}

