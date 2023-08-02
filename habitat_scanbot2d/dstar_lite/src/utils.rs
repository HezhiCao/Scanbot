use ndarray::ArrayView1;
use std::cmp;

/// rhs & g data for A* related searches
#[derive(Debug, Clone, Copy)]
pub struct NodeState {
    pub rhs: f64,
    pub g: f64,
}

#[derive(Debug, Hash, PartialEq, Eq, Default, Clone, Copy)]
pub struct NodePosition(pub usize, pub usize);

impl NodePosition {
    pub fn from(array: ArrayView1<i32>) -> Self {
        Self(array[0] as usize, array[1] as usize)
    }
}

/// vertex stored in priority queue
#[derive(Debug)]
pub struct HeapEntry {
    pub position: NodePosition,
    pub keys: (f64, f64),
}

impl HeapEntry {
    pub fn new(position: NodePosition, keys: (f64, f64)) -> Self {
        Self { position, keys }
    }

    #[inline]
    fn transform_keys(&self) -> (i64, i64) {
        let key0 = (self.keys.0 * 1024.0 * 1024.0).round() as i64;
        let key1 = (self.keys.1 * 1024.0 * 1024.0).round() as i64;
        (key0, key1)
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        let (self_key0, self_key1) = self.transform_keys();
        let (other_key0, other_key1) = other.transform_keys();
        (self_key0, self_key1) == (other_key0, other_key1)
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        // First compare key0 then key1
        let (self_key0, self_key1) = self.transform_keys();
        let (other_key0, other_key1) = other.transform_keys();
        // rust default BinaryHeap is a max-heap
        // So we need reverse the comparing result
        other_key0
            .cmp(&self_key0)
            .then(other_key1.cmp(&self_key1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    #[test]
    fn test_queue_entries_ordering() {
        let mut priority_queue = BinaryHeap::new();
        let pos1 = NodePosition(1, 1);
        let pos2 = NodePosition(2, 3);
        let pos3 = NodePosition(3, 4);
        priority_queue.push(HeapEntry::new(pos1, (1.1, 2.0)));
        priority_queue.push(HeapEntry::new(pos2, (2.5, 2.0)));
        priority_queue.push(HeapEntry::new(pos3, (1.1, 3.0)));

        assert_eq!(priority_queue.peek().unwrap().position, pos1);
        assert_eq!(priority_queue.pop().unwrap().keys, (1.1, 2.0));
        assert_eq!(priority_queue.pop().unwrap().keys, (1.1, 3.0));
        assert_eq!(priority_queue.pop().unwrap().keys, (2.5, 2.0));
    }
}
