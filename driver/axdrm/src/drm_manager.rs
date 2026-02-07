
use alloc::{collections::btree_map::BTreeMap, vec::Vec};
use spin::Mutex;
use lazy_static::lazy_static;
const MAX_GEM_OBJECTS: usize = 1024; // 最大支持的gem对象缓存数量

lazy_static!{
    pub static ref GEM_ALLOCATOR: Mutex<GemNumberAllocator> = Mutex::new(GemNumberAllocator::new());
}


pub struct GemNumberAllocator {
    pub current: u32, // 当前分配的gem对象数量
    pub max: u32, // 最大分配的gem对象数量
    pub map: BTreeMap<GemHandle, GemObject>, // handle到GemObject的映射
    pub free_list: Vec<GemHandle>, // 空闲的gemHandle
}


impl GemNumberAllocator {
    pub fn new() -> Self {
        Self {
            current: 0,
            max: MAX_GEM_OBJECTS as u32,
            map: BTreeMap::new(),
            free_list: Vec::new(),
        }
    }

    /// 分配一个 GEM handle 编号，优先从 free_list 回收，否则递增
    pub fn allocate(&mut self) -> Option<GemHandle> {
        // 优先从空闲列表回收
        if !self.free_list.is_empty() {
            return self.free_list.pop();
        }
        // 没有可回收的，递增分配
        if self.current >= self.max {
            return None; // 已达上限
        }
        self.current += 1;
        let handle = Some(GemHandle(self.current));

        // TODO:
        // 创建一个新的 GemObject 并加入映射
        // 向os事情dma分配物理内存，获取物理地址和虚拟地址
        


        return handle;
    }

    /// 回收一个 GEM handle 编号，放回 free_list
    pub fn free(&mut self, handle: u32) {
        self.free_list.push(GemHandle(handle));
        self.map.remove(&GemHandle(handle)); // 同时移除映射
    }
}


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct GemHandle(u32); // 手牌号

pub struct GemObject {
    pub handle: GemHandle, // 手牌号
    pub phy_addr: u64, // 物理地址
    pub size: usize, // 大小
    pub fake_offset: usize, // 假偏移地址,给mmap看的
    pub vaddr: usize, // 虚拟地址 (内核视角)
}

impl Drop for GemHandle {
    fn drop(&mut self) {
        // 回收 handle 编号
        GEM_ALLOCATOR.lock().free(self.0);
    }
}


