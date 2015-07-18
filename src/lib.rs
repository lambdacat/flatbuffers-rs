// Copyright 2015 Sam Payson. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

extern crate num;

use std::cmp::Eq;

use std::cmp;
use std::marker;
use std::mem;
use std::slice;
use std::str;

// Return a byte slice which refers to the same region of memory as `v`.
fn view_slice_bytes<T>(v: &[T]) -> &[u8] {
    unsafe {
        let ptr = mem::transmute::<&T, *const u8>(&v[0]);
        let len = mem::size_of::<T>() * v.len();

        slice::from_raw_parts(ptr, len)
    }
}

// Return a byte slice which refers to the same region of memory as `t`.
fn view_bytes<T>(t: &T) -> &[u8] {
    unsafe {
        let ptr = mem::transmute::<&T, *const u8>(t);
        let len = mem::size_of::<T>();

        slice::from_raw_parts(ptr, len)
    }
}

/// An unsigned offset
pub type UOffset = u32;

/// A signed offset
pub type SOffset = i32;

/// A vtable offset, used for indexing the fields of a Table
pub type VOffset = u16;

/// This is a trait for primitives which can be loaded and stored as aligned little-endian values.
pub trait Endian: Copy + PartialEq {
    unsafe fn read_le(buf: *const u8) -> Self;
    unsafe fn write_le(self, buf: *mut u8);

    fn from_le(self) -> Self;
    fn to_le(self) -> Self;
}

// What we really want here is:
//
//     impl<T: num::PrimInt> Endian for T {
//         fn read_le(buf: &[u8]) -> T {
//             let ptr: &T = unsafe { mem::transmute(&buf[0]) };
//             num::PrimInt::from_le(*ptr)
//         }
//
//         fn write_le(self, buf: &mut [u8]) {
//             let ptr: &mut T = unsafe { mem::transmute(&mut buf[0]) };
//             *ptr = self.to_le();
//         }
//     }
//
// but the blanket impl causes errors if we try to implement it for any other type, so this macro
// will have to do.
macro_rules! impl_endian_for {
    ($t:ty) => {
        impl Endian for $t {
            unsafe fn read_le(buf: *const u8) -> $t {
                let ptr = mem::transmute::<*const u8, &$t>(buf);
                num::PrimInt::from_le(*ptr)
            }

            unsafe fn write_le(self, buf: *mut u8) {
                let ptr = mem::transmute::<*mut u8, &mut $t>(buf);
                *ptr = num::PrimInt::to_le(self);
            }

            fn from_le(self) -> $t { num::PrimInt::from_le(self) }

            fn to_le(self) -> $t { num::PrimInt::to_le(self) }
        }
    }
}

impl_endian_for!(u8);
impl_endian_for!(i8);
impl_endian_for!(u16);
impl_endian_for!(i16);
impl_endian_for!(u32);
impl_endian_for!(i32);
impl_endian_for!(u64);
impl_endian_for!(i64);
impl_endian_for!(usize);
impl_endian_for!(isize);

/// This implementation assumes that the endianness of the FPU is the same as for integers.
impl Endian for f32 {
    fn from_le(self) -> f32 {
        unsafe {
            let u = mem::transmute::<f32, u32>(self);
            mem::transmute::<u32, f32>(num::PrimInt::from_le(u))
        }
    }

    fn to_le(self) -> f32 {
        unsafe {
            let u = mem::transmute::<f32, u32>(self);
            mem::transmute::<u32, f32>(num::PrimInt::to_le(u))
        }
    }

    unsafe fn read_le(buf: *const u8) -> f32 {
        let ptr = mem::transmute::<*const u8, &u32>(buf);
        mem::transmute::<u32, f32>(num::PrimInt::from_le(*ptr))
    }

    unsafe fn write_le(self, buf: *mut u8) {
        let ptr = mem::transmute::<*mut u8, &mut u32>(buf);
        *ptr = num::PrimInt::to_le(mem::transmute::<f32, u32>(self));
    }
}

/// This implementation assumes that the endianness of the FPU is the same as for integers.
impl Endian for f64 {
    fn from_le(self) -> f64 {
        unsafe {
            let u = mem::transmute::<f64, u64>(self);
            mem::transmute::<u64, f64>(num::PrimInt::from_le(u))
        }
    }

    fn to_le(self) -> f64 {
        unsafe {
            let u = mem::transmute::<f64, u64>(self);
            mem::transmute::<u64, f64>(num::PrimInt::to_le(u))
        }
    }

    unsafe fn read_le(buf: *const u8) -> f64 {
        let ptr = mem::transmute::<*const u8, &u64>(buf);
        mem::transmute::<u64, f64>(num::PrimInt::from_le(*ptr))
    }

    unsafe fn write_le(self, buf: *mut u8) {
        let ptr = mem::transmute::<*mut u8, &mut u64>(buf);
        *ptr = num::PrimInt::to_le(mem::transmute::<f64, u64>(self));
    }
}

impl Endian for bool {
    fn from_le(self) -> bool { self }

    fn to_le(self) -> bool { self }

    unsafe fn read_le(buf: *const u8) -> bool {
        let ptr = mem::transmute::<*const u8, &bool>(buf);
        *ptr
    }

    unsafe fn write_le(self, buf: *mut u8) {
        let ptr = mem::transmute::<*mut u8, &mut bool>(buf);
        *ptr = self;
    }
}

impl<T> Endian for Offset<T> {
    fn from_le(self) -> Offset<T> {
        Offset::new(num::PrimInt::from_le(self.inner))
    }

    fn to_le(self) -> Offset<T> {
        Offset::new(num::PrimInt::to_le(self.inner))
    }

    unsafe fn read_le(buf: *const u8) -> Offset<T> {
        let ptr = mem::transmute::<*const u8, &UOffset>(buf);
        Offset::new(num::PrimInt::from_le(*ptr))
    }

    unsafe fn write_le(self, buf: *mut u8) {
        let ptr = mem::transmute::<*mut u8, &mut UOffset>(buf);
        *ptr = num::PrimInt::to_le(self.inner)
    }
}

// If `base` were a pointer to an array of type T, return a pointer to element `idx` of that array.
unsafe fn index<T>(base: *const u8, idx: usize) -> *const u8 {
    let base_us = mem::transmute::<*const u8, usize>(base);

    mem::transmute::<usize, *const u8>(base_us + idx * mem::size_of::<T>())
}

// Return a pointer to a byte whose address is `off` bytes beyond `base`.
unsafe fn offset(base: *const u8, off: usize) -> *const u8 {
    let base_us = mem::transmute::<*const u8, usize>(base);

    mem::transmute::<usize, *const u8>(base_us + off)
}

// This is like `offset`, except it returns a mutable pointer.
unsafe fn offset_mut(base: *mut u8, off: usize) -> *mut u8 {
    let base_us = mem::transmute::<*mut u8, usize>(base);

    mem::transmute::<usize, *mut u8>(base_us + off)
}

// This is like `offset` except it allows negative offsets.
unsafe fn soffset(base: *const u8, off: isize) -> *const u8 {
    let base_is = mem::transmute::<*const u8, isize>(base);

    mem::transmute::<isize, *const u8>(base_is + off)
}

// Read a little endian `T` pointed to by `buf`. `buf` must point to a properly aligned value.
unsafe fn read_scalar<T: Endian>(buf: *const u8) -> T {
    Endian::read_le(buf)
}

// Write a little endian `T` to the buffer pointer to by `buf`. `buf` must point to a properly
// aligned buffer.
unsafe fn write_scalar<T: Endian>(buf: *mut u8, val: T) {
    val.write_le(buf)
}

/// A trait which determines how a type is retrieved from a flatbuffer. See the implementations for
/// `T`, `Offset<T>`, and `ByRef<T>` for examples.
pub trait Indirect<I> {
    unsafe fn read(buf: *const u8, idx: usize) -> I;
}

impl<T: Copy> Indirect<T> for T {
    unsafe fn read(buf: *const u8, idx: usize) -> T {
        let off = idx * mem::size_of::<T>();
        let ptr = mem::transmute::<*const u8, &T>(offset(buf, off));

        *ptr
    }
}

pub struct Offset<T> {
    inner: UOffset,
    _t:    marker::PhantomData<T>,
}

impl<T> Clone for Offset<T> {
    fn clone(&self) -> Self { Offset::new(self.inner) }
}

impl<T> Copy for Offset<T> {}

impl<T> PartialEq for Offset<T> {
    fn eq(&self, other: &Self) -> bool { self.inner == other.inner }
}

pub struct ByRef<T>(marker::PhantomData<T>);

impl <'x, T> Indirect<&'x T> for ByRef<T> {
    unsafe fn read(buf: *const u8, idx: usize) -> &'x T {
        mem::transmute::<*const u8, &'x T>(index::<T>(buf, idx))
    }
}

impl<'x, T> Indirect<&'x T> for Offset<T> {
    unsafe fn read(buf: *const u8, idx: usize) -> &'x T {
        let off: UOffset = read_scalar(index::<UOffset>(buf, idx));
        mem::transmute::<*const u8, &'x T>(offset(buf, off as usize))
    }
}

impl<T> Offset<T> {
    pub fn new(o: UOffset) -> Offset<T> {
        Offset {
            inner: o,
            _t:    marker::PhantomData,
        }
    }
}

/// A helper type for accessing vectors in flatbuffers.
pub struct Vector<T, I = T> where T: Indirect<I> {
    length: UOffset,
    _t:     marker::PhantomData<T>,
    _i:     marker::PhantomData<I>,
}

/// An iterator to a Vector in a flatbuffer.
pub struct VecIter<'x, I: 'x, T: Indirect<I> + 'x> {
    vec: &'x Vector<T, I>,
    idx: usize,
}

impl<'x, I, T: Indirect<I>> Iterator for VecIter<'x, I, T> {
    type Item = I;

    fn next(&mut self) -> Option<I> {
        let idx = self.idx;
        self.idx = idx + 1;

        self.vec.get(idx)
    }
}

impl<I, T: Indirect<I>> Vector<T, I> {
    unsafe fn data(&self) -> *const u8 {
        index::<UOffset>(mem::transmute::<&Vector<T, I>, *const u8>(self), 1)
    }

    pub fn len(&self) -> usize {
        self.length as usize
    }

    pub fn get(&self, idx: usize) -> Option<I> {
        if idx < self.len() {
            Some(unsafe { <T as Indirect<I>>::read(self.data(), idx) })
        } else {
            None
        }
    }

    pub fn iter<'x>(&'x self) -> VecIter<'x, I, T> {
        VecIter {
            vec: self,
            idx: 0,
        }
    }
}

pub type Str = Vector<i8>;

impl AsRef<str> for Str {
    fn as_ref(&self) -> &str {
        let slc = unsafe {
            let ptr = self.data();
            let len = self.len();

            slice::from_raw_parts(ptr, len)
        };

        // TODO: Should this be the checked version? If so, do we want to panic if it's not utf-8?
        //
        //       This (unchecked) version certainly reflects the performance characteristics in the
        //       spirit of the format. Maybe the `AsRef<str>` implementation should be checked, and
        //       there should be an unsafe fast method?
        //
        //       I'll think about it later...
        unsafe { str::from_utf8_unchecked(slc) }
    }
}

impl PartialEq for Str {
    fn eq(&self, other: &Str) -> bool {
        let (a, b): (&str, &str) = (self.as_ref(), other.as_ref());
        a.eq(b)
    }
}

impl PartialOrd for Str {
    fn partial_cmp(&self, other: &Str) -> Option<cmp::Ordering> {
        let (a, b): (&str, &str) = (self.as_ref(), other.as_ref());
        a.partial_cmp(b)
    }
}

impl Eq for Str {}

impl Ord for Str {
    fn cmp(&self, other: &Str) -> cmp::Ordering {
        let (a, b): (&str, &str) = (self.as_ref(), other.as_ref());
        a.cmp(b)
    }
}

pub struct Table;

impl Table {
    fn get_optional_field_offset(&self, field: VOffset) -> Option<VOffset> {
        unsafe {
            let base = mem::transmute::<&Table, *const u8>(self);

            // I'm not suire why it's subtraction, instead of addition, but this is what they have in
            // the C++ code.
            let vtable = soffset(base, -read_scalar::<SOffset>(base) as isize);

            let vtsize: VOffset = read_scalar(vtable);

            if field < vtsize {
                let voff = read_scalar(offset(vtable, field as usize));
                if voff != 0 {
                    return Some(voff)
                }
            }

            None
        }
    }

    pub fn get_field<T: Endian>(&self, field: VOffset, def: T) -> T {

        self.get_optional_field_offset(field)
            .map_or(def, |voffs| unsafe {
                let base = mem::transmute::<&Table, *const u8>(self);
                read_scalar(offset(base, voffs as usize))
            } )
    }

    pub fn get_ref<T>(&self, field: VOffset) -> Option<&T> {
        self.get_optional_field_offset(field)
            .map(|voffs| unsafe {
                let base = mem::transmute::<&Table, *const u8>(self);
                let p    = offset(base, voffs as usize);
                let offs: UOffset = read_scalar(p);
                mem::transmute::<*const u8, &T>(offset(p, offs as usize))
            })
    }

    pub fn get_ref_mut<T>(&mut self, field: VOffset) -> Option<&mut T> {
        self.get_optional_field_offset(field)
            .map(|voffs| unsafe {
                let base = mem::transmute::<&mut Table, *mut u8>(self);
                let p    = offset_mut(base, voffs as usize);
                let offs: UOffset = read_scalar(p);
                mem::transmute::<*mut u8, &mut T>(offset_mut(p, offs as usize))
            })
    }

    pub fn get_struct<T>(&self, field: VOffset) -> Option<&T> {
        self.get_optional_field_offset(field)
            .map(|voffs| unsafe {
                let base = mem::transmute::<&Table, *const u8>(self);
                mem::transmute::<*const u8, &T>(offset(base, voffs as usize))
            })
    }

    pub fn get_struct_mut<T>(&mut self, field: VOffset) -> Option<&mut T> {
        self.get_optional_field_offset(field)
            .map(|voffs| unsafe {
                let base = mem::transmute::<&mut Table, *mut u8> (self);
                mem::transmute::<*mut u8, &mut T>(offset_mut(base, voffs as usize))
            })
    }

    pub fn set_field<T: Endian>(&mut self, field: VOffset, val: T) {
        unsafe {
            // We `unwrap` here because the caller is expected to verify that the field exists
            // beforehand by calling `check_field`.
            let voffs = self.get_optional_field_offset(field).unwrap();

            let base = mem::transmute::<&mut Table, *mut u8>(self);

            write_scalar(offset_mut(base, voffs as usize), val);
        }
    }

    pub fn check_field(&self, field: VOffset) -> bool {
        self.get_optional_field_offset(field).is_some()
    }
}

/// A trait for Tables which can be compared for order (i.e. which have a field with the `key`
/// attribute).
pub trait OrdTable {
    fn key_cmp(&self, rhs: &Self) -> cmp::Ordering;
}

/// This type is used internally by the generated types for flatbuffer structs. Its methods allow
/// access to various different types of struct fields.
pub struct Struct;

impl Struct {
    /// Return a scalar field, reading it directly from the buffer.
    pub fn get_field<T: Endian>(&self, off: UOffset) -> T {
        unsafe {
            let base = mem::transmute::<&Struct, *const u8>(self);
            read_scalar(offset(base, off as usize))
        }
    }

    /// Return a reference to a field which is stored as a `UOffset`.
    ///
    /// # Notes
    ///
    /// Is this function ever used? Aren't structs supposed to be fixed-size?
    pub fn get_ref<T>(&self, off: UOffset) -> &T {
        unsafe {
            let base = mem::transmute::<&Struct, *const u8>(self);
            let p    = offset(base, off as usize);

            mem::transmute::<*const u8, &T>(offset(p, read_scalar::<UOffset>(p) as usize))
        }
    }

    /// Like `get_ref`, but the reference is mutable.
    pub fn get_ref_mut<T>(&mut self, off: UOffset) -> &mut T {
        unsafe {
            let base = mem::transmute::<&mut Struct, *mut u8>(self);
            let p    = offset_mut(base, off as usize);

            mem::transmute::<*mut u8, &mut T>(offset_mut(p, read_scalar::<UOffset>(p) as usize))
        }
    }

    /// Get a pointer to a struct field.
    pub fn get_struct<T>(&self, off: UOffset) -> &T {
        unsafe {
            let base = mem::transmute::<&Struct, *const u8>(self);

            mem::transmute::<*const u8, &T>(offset(base, off as usize))
        }
    }

    /// Like `get_struct`, but the reference is mutable.
    pub fn get_struct_mut<T>(&mut self, off: UOffset) -> &mut T {
        unsafe {
            let base = mem::transmute::<&mut Struct, *mut u8>(self);

            mem::transmute::<*mut u8, &mut T>(offset_mut(base, off as usize))
        }
    }
}

/// Return a pointer to the root object stored in this buffer, interpreting it as type `T`.
pub fn get_root<T>(buf: &[u8]) -> &T {
    unsafe {
        let base         = buf.as_ptr();
        let off: UOffset = Endian::read_le(base);

        mem::transmute::<*const u8, &T>(offset(base, off as usize))
    }
}

// Reverse-growing vector which piggy-backs on std::vec::Vec.
struct VecDownward {
    inner: Vec<u8>,
    next:  usize,
}

impl VecDownward {
    fn new(initial_capacity: usize) -> VecDownward {
        let mut vec = Vec::with_capacity(initial_capacity);
        unsafe { vec.set_len(initial_capacity) }

        VecDownward {
            inner: vec,
            next:  initial_capacity,
        }
    }

    fn data(&self) -> &[u8] { &self.inner[self.next..] }

    fn data_mut(&mut self) -> &mut [u8] { &mut self.inner[self.next..] }

    // TODO: Implement Index and IndexMut traits?
    fn data_at(&self, offset: usize) -> &[u8] {
        &self.inner[self.inner.len() - offset..]
    }

    fn data_at_mut(&mut self, offset: usize) -> &mut [u8] {
        let len = self.inner.len();
        &mut self.inner[len - offset..]
    }

    fn len(&self) -> usize { self.inner.len() - self.next }

    fn clear(&mut self) {
        self.next = self.inner.len();
    }

    // Adds space to the front of the vector, growing towards lower addresses. The returned `usize`
    // is the offset from the end of the buffer (e.g. the highest address).
    fn make_space(&mut self, len: usize) -> usize {
        if len > self.next {
            let mut new = Vec::with_capacity(2*self.len() + len);

            unsafe { new.set_len(2*self.len() + len) }

            let new_next = new.len() - self.len();

            for i in 0..self.len() {
                new[new_next + i] = self.inner[self.next + i]
            }

            self.inner = new;
            self.next = new_next;
        }

        self.next -= len;

        self.next
    }

    // Append some raw bytes to the front of the buffer.
    fn push(&mut self, dat: &[u8]) {
        let off = self.make_space(dat.len());

        for i in 0..dat.len() {
            self.inner[off + i] = dat[i];
        }
    }

    // Add `len` *NUL* bytes to the front of the buffer.
    fn fill(&mut self, len: usize) {
        let off = self.make_space(len);

        for i in 0..len {
            self.inner[off + i] = 0;
        }
    }

    // Remove `len` bytes from the front of the buffer.
    fn pop(&mut self, len: usize) {
        self.next += len;
    }
}

// Given a field's ID number, convert it to a VOffset
fn field_index_to_offset(field_id: VOffset) -> VOffset {
    let fixed_fields = 2; // VTable size and Object size.
    (field_id + fixed_fields) * (mem::size_of::<VOffset>() as VOffset)
}

// Return the number of bytes needed to pad a scalar for alignment (see usage in e.g.
// `FlatBufferBuilder::align(..)`).
fn padding_bytes(buf_size: usize, scalar_size: usize) -> usize {
    (!buf_size).wrapping_add(1) & (scalar_size - 1)
}

// The location of a field, stored as a UOffset from the end of the buffer and a field ID.
struct FieldLoc {
    off: UOffset,
    id:  VOffset,
}

/// This type is used by the generated `.*Builder` types for Tables. A `FlatBufferBuilder` can be
/// re-used if the `clear()` method is called between uses; this will avoid some allocations.
pub struct FlatBufferBuilder {
    buf:            VecDownward,
    offset_buf:     Vec<FieldLoc>,
    vtables:        Vec<UOffset>,
    min_align:      usize,
    force_defaults: bool,
}

impl FlatBufferBuilder {
    pub fn new(initial_capacity: usize) -> FlatBufferBuilder {
        FlatBufferBuilder {
            buf:            VecDownward::new(initial_capacity),
            offset_buf:     Vec::with_capacity(16),
            vtables:        Vec::with_capacity(16),
            min_align:      1,
            force_defaults: false,
        }
    }

    /// Prepare to build another FlatBuffer from scratch (forgetting everything about any previous
    /// FlatBuffer), but reuse the memory from the internal buffers to avoid extra reallocations.
    pub fn clear(&mut self) {
        self.buf.clear();
        self.offset_buf.clear();
        self.vtables.clear();
        self.min_align = 1;
    }

    pub fn get_size(&self) -> usize {
        self.buf.len()
    }

    pub fn get_buffer(&self) -> &[u8] { self.buf.data() }

    /// Determines whether or not default values should be hard-coded into the wire representation.
    pub fn force_defaults(&mut self, fd: bool) {
        self.force_defaults = fd;
    }

    pub fn pad(&mut self, num_bytes: usize) {
        self.buf.fill(num_bytes);
    }

    pub fn align(&mut self, elem_size: usize) {
        if elem_size > self.min_align {
            self.min_align = elem_size;
        }

        let len = self.buf.len();

        self.buf.fill(padding_bytes(len, elem_size));
    }

    pub fn push_bytes(&mut self, dat: &[u8]) {
        self.buf.push(dat);
    }

    pub fn pop_bytes(&mut self, len: usize) {
        self.buf.pop(len)
    }

    pub fn push_scalar<T: Endian>(&mut self, elem: T) -> usize {
        let little = elem.to_le();

        self.align(mem::size_of::<T>());

        self.buf.push(view_bytes(&little));

        self.get_size()
    }

    pub fn push_offset<T>(&mut self, off: Offset<T>) -> usize {
        let adjusted = self.refer_to(off.inner);
        self.push_scalar(adjusted)
    }

    pub fn refer_to(&mut self, off: UOffset) -> UOffset {
        self.align(mem::size_of::<UOffset>());
        let buf_size = self.get_size() as UOffset;

        assert!(off <= buf_size);

        buf_size - off + (mem::size_of::<UOffset>() as UOffset)
    }

    pub fn track_field(&mut self, field: VOffset, off: UOffset) {
        // TODO: Make field an index and compute offset with field_index_to_offset.
        self.offset_buf.push(FieldLoc{off: off, id: field})
    }

    pub fn add_scalar<T: Endian>(&mut self, field: VOffset, e: T, def: T) {
        if e == def && !self.force_defaults { return }

        let off = self.push_scalar(e) as UOffset;

        self.track_field(field, off);
    }

    pub fn add_offset<T>(&mut self, field: VOffset, off: Offset<T>) {
        if off.inner == 0 { return }

        let adjusted = self.refer_to(off.inner);
        self.add_scalar(field, adjusted, 0);
    }

    pub fn add_struct<T>(&mut self, field: VOffset, ptr: &T) {
        self.align(mem::align_of::<T>());
        self.push_bytes(view_bytes(ptr));

        let off = self.get_size() as UOffset;
        self.track_field(field, off);
    }

    pub fn add_struct_offset(&mut self, field: VOffset, off: UOffset) {
        self.track_field(field, off);
    }

    pub fn not_nested(&self) {
        assert_eq!(self.offset_buf.len(), 0);
    }

    pub fn start_table(&self) -> UOffset {
        self.not_nested();
        self.get_size() as UOffset
    }

    pub fn end_table(&mut self, start: UOffset, num_fields: VOffset) -> UOffset {
        let vtable_offset_loc = self.push_scalar(0 as SOffset);

        self.buf.fill((num_fields as usize) * mem::size_of::<VOffset>());

        let table_object_size = vtable_offset_loc - (start as usize);

        assert!(table_object_size < 0x10000); // 16-bit offsets

        self.push_scalar(table_object_size as VOffset);
        self.push_scalar(field_index_to_offset(num_fields));

        for field_loc in &self.offset_buf {
            let pos = (vtable_offset_loc as u32 - field_loc.off) as VOffset;

            unsafe {
                let buf_ref = &mut self.buf.data_mut()[field_loc.id as usize];
                let buf_ptr = mem::transmute::<&mut u8, *mut u8>(buf_ref);
                assert_eq!(read_scalar::<VOffset>(buf_ptr), 0);
                write_scalar(buf_ptr, pos);
            }
        }

        self.offset_buf.clear();

        // What follows is the de-duping code. Might be able to speed this up with some kind of
        // hash-table or something if it becomes a bottleneck since this implementation will take
        // quadratic time WRT the number of distinct tables in the flatbuffer.

        // TODO: Can we get a VOffset instead?
        let vt1: &[u8] = unsafe {
            let vt_ptr = mem::transmute::<&u8, *const u8>(&self.buf.data()[0]);
            let vt_len = *vt_ptr as usize;
            slice::from_raw_parts(vt_ptr, vt_len)
        };

        let mut vt_use = self.get_size() as UOffset;

        for &off in self.vtables.iter() {
            // TODO: Can we get a VOffset instead?
            let vt2: &[u8] = unsafe {
                let vt_ptr = mem::transmute::<&u8, *const u8>(&self.buf.data_at(off as usize)[0]);
                let vt_len = *vt_ptr as usize;
                slice::from_raw_parts(vt_ptr, vt_len)
            };

            if vt1 == vt2 {
                vt_use = off;
                let to_pop = self.get_size() - vtable_offset_loc;
                self.buf.pop(to_pop);
                break;
            }
        }

        if vt_use == self.get_size() as UOffset {
            self.vtables.push(vt_use);
        }

        unsafe {
            let vt_buf = &mut self.buf.data_at_mut(vtable_offset_loc as usize)[0];
            write_scalar(mem::transmute::<*mut u8, &mut u8>(vt_buf),
                (vt_use as SOffset) - (vtable_offset_loc as SOffset));
        }

        vtable_offset_loc as UOffset
    }

    pub fn pre_align(&mut self, len: usize, align: usize) {
        let size = self.get_size();
        self.buf.fill(padding_bytes(size + len, align));
    }

    pub fn create_string(&mut self, s: &str) -> Offset<Str> {
        self.not_nested();

        self.pre_align(s.len() + 1, mem::size_of::<UOffset>());
        self.buf.fill(1);
        self.push_bytes(s.as_bytes());
        self.push_scalar(s.len() as UOffset);

        Offset::new(self.get_size() as UOffset)
    }

    pub fn start_vector(&mut self, len: usize, elem_size: usize) {
        self.pre_align(len * elem_size, mem::size_of::<UOffset>());
        self.pre_align(len * elem_size, elem_size);
    }

    pub fn reserve_elements(&mut self, len: usize, elem_size: usize) -> usize {
        self.buf.make_space(len * elem_size)
    }

    pub fn end_vector(&mut self, len: usize) -> UOffset {
        self.push_scalar(len as UOffset) as UOffset
    }

    pub fn create_vector<T: Endian>(&mut self, v: &[T]) -> Offset<Vector<T>> {
        self.not_nested();
        self.start_vector(v.len(), mem::size_of::<T>());
        for &elem in v.iter().rev() {
            self.push_scalar(elem);
        }

        Offset::new(self.end_vector(v.len()))
    }

    pub fn create_vector_of_structs<T>(&mut self, v: &[T]) -> Offset<Vector<ByRef<T>, &T>> {
        self.not_nested();

        self.start_vector(v.len() * mem::size_of::<T>() / mem::align_of::<T>(),
                          mem::align_of::<T>());
        self.push_bytes(view_slice_bytes(v));

        Offset::new(self.end_vector(v.len()))
    }

    pub fn create_vector_of_sorted_tables<T: OrdTable>(&mut self, v: &mut [Offset<T>])
        -> Offset<Vector<Offset<T>>> {

        v.sort_by(|&a_off, &b_off| {
            unsafe {
                let a = mem::transmute::<&u8, &T>(&self.buf.data_at(a_off.inner as usize)[0]);
                let b = mem::transmute::<&u8, &T>(&self.buf.data_at(b_off.inner as usize)[0]);

                a.key_cmp(b)
            }
        });

        self.create_vector(v)
    }

    pub fn create_uninitialized_vector<T>(&mut self, len: usize) -> (UOffset, &mut [T]) {
        self.not_nested();
        self.start_vector(len, mem::size_of::<T>());
        let buf = self.buf.make_space(len * mem::size_of::<T>());
        let off = self.end_vector(len);

        let slc = unsafe {
            let ptr = mem::transmute::<&mut u8, *mut T>(&mut self.buf.data_at_mut(buf)[0]);

            slice::from_raw_parts_mut(ptr, len)
        };

        (off, slc)
    }

    pub fn finish<T>(&mut self, root: Offset<T>) {
        let min_align = self.min_align;
        self.pre_align(mem::size_of::<UOffset>(), min_align);
        let refer = self.refer_to(root.inner);
        self.push_scalar(refer);
    }
}

#[cfg(test)]
mod test {
    use std::mem;

    use super::*;

    #[test]
    fn test_numbers() {
        let mut builder = FlatBufferBuilder::new(0);
        assert_eq!(builder.get_buffer(), []);
        builder.push_scalar(true);
        assert_eq!(builder.get_buffer(), [1]);
        builder.push_scalar(-127i8);
        assert_eq!(builder.get_buffer(), [129, 1]);
        builder.push_scalar(255u8);
        assert_eq!(builder.get_buffer(), [255, 129, 1]);
        builder.push_scalar(-32222i16);
        assert_eq!(builder.get_buffer(), [0x22, 0x82, 0, 255, 129, 1]);
        builder.push_scalar(0xFEEEu16);
        assert_eq!(builder.get_buffer(), [0xEE, 0xFE, 0x22, 0x82, 0, 255, 129, 1]);
        builder.push_scalar(-53687092i32);
        assert_eq!(builder.get_buffer(), [204, 204, 204, 252, 0xEE, 0xFE, 0x22,
                                          0x82, 0, 255, 129, 1]);
        builder.push_scalar(0x98765432u32);
        assert_eq!(builder.get_buffer(), [0x32, 0x54, 0x76, 0x98,
                                          204, 204, 204, 252,
                                          0xEE, 0xFE, 0x22, 0x82,
                                          0, 255, 129, 1]);
    }

    #[test]
    fn test_numbers64() {
        let mut builder = FlatBufferBuilder::new(0);
        builder.push_scalar(0x1122334455667788u64);
        assert_eq!(builder.get_buffer(), [0x88, 0x77, 0x66, 0x55,
                                          0x44, 0x33, 0x22, 0x11]);

        builder = FlatBufferBuilder::new(0);
        builder.push_scalar(0x1122334455667788i64);
        assert_eq!(builder.get_buffer(), [0x88, 0x77, 0x66, 0x55,
                                          0x44, 0x33, 0x22, 0x11]);
    }

    #[test]
    fn test_1xbyte_vector() {
        let mut builder = FlatBufferBuilder::new(0);
        assert_eq!(builder.get_buffer(), []);
        builder.start_vector(1, mem::size_of::<u8>());
        assert_eq!(builder.get_buffer(), [0, 0, 0]);
        builder.push_scalar(1u8);
        assert_eq!(builder.get_buffer(), [1, 0, 0, 0]);
        builder.end_vector(1);
        assert_eq!(builder.get_buffer(), [1, 0, 0, 0, 1, 0, 0, 0]);
    }

    #[test]
    fn test_2xbyte_vector() {
        let mut builder = FlatBufferBuilder::new(0);
        assert_eq!(builder.get_buffer(), []);
        builder.start_vector(2, mem::size_of::<u8>());
        assert_eq!(builder.get_buffer(), [0, 0]);
        builder.push_scalar(1u8);
        assert_eq!(builder.get_buffer(), [1, 0, 0]);
        builder.push_scalar(2u8);
        assert_eq!(builder.get_buffer(), [2, 1, 0, 0]);
        builder.end_vector(2);
        assert_eq!(builder.get_buffer(), [2, 0, 0, 0, 2, 1, 0, 0]);
    }

    #[test]
    fn test_1xuint16_vector() {
        let mut builder = FlatBufferBuilder::new(0);
        assert_eq!(builder.get_buffer(), []);
        builder.start_vector(1, mem::size_of::<u16>());
        assert_eq!(builder.get_buffer(), [0, 0]);
        builder.push_scalar(1u16);
        assert_eq!(builder.get_buffer(), [1, 0, 0, 0]);
        builder.end_vector(1);
        assert_eq!(builder.get_buffer(), [1, 0, 0, 0, 1, 0, 0, 0]);
    }

    #[test]
    fn test_2xuint16_vector() {
        let mut builder = FlatBufferBuilder::new(0);
        assert_eq!(builder.get_buffer(), []);
        builder.start_vector(2, mem::size_of::<u16>());
        assert_eq!(builder.get_buffer(), []);
        builder.push_scalar(0xABCDu16);
        assert_eq!(builder.get_buffer(), [0xCD, 0xAB]);
        builder.push_scalar(0xDCBAu16);
        assert_eq!(builder.get_buffer(), [0xBA, 0xDC, 0xCD, 0xAB]);
        builder.end_vector(2);
        assert_eq!(builder.get_buffer(), [2, 0, 0, 0, 0xBA, 0xDC, 0xCD, 0xAB]);
    }

    #[test]
    fn test_create_ascii_string() {
        let mut builder = FlatBufferBuilder::new(0);
        builder.create_string("foo"); // TODO: Pass in a byte string?
        assert_eq!(builder.get_buffer(), [3, 0, 0, 0, b'f', b'o', b'o', 0]);
        builder.create_string("moop"); // TODO: Pass in a byte string?
        assert_eq!(builder.get_buffer(), [4, 0, 0, 0, b'm', b'o', b'o', b'p',
                                          0, 0, 0, 0,
                                          3, 0, 0, 0, b'f', b'o', b'o', 0]);
    }

    #[test]
    fn test_create_arbitrary_string() {
        let mut builder = FlatBufferBuilder::new(0);
        builder.create_string("\x01\x02\x03");
        assert_eq!(builder.get_buffer(), [3, 0, 0, 0, 1, 2, 3, 0]);
        builder.create_string("\x04\x05\x06\x07");
        assert_eq!(builder.get_buffer(), [4, 0, 0, 0, 4, 5, 6, 7,
                                          0, 0, 0, 0,
                                          3, 0, 0, 0, 1, 2, 3, 0]);
    }

    #[test]
    fn test_empty_vtable() {
        let mut builder = FlatBufferBuilder::new(0);
        let start_offset = builder.start_table();
        assert_eq!(builder.get_buffer(), []);
        builder.end_table(start_offset, 0);
        assert_eq!(builder.get_buffer(), [4, 0, 4, 0, 4, 0, 0, 0]);
    }

    #[test]
    fn test_vtable_with_one_true_bool() {
        let mut builder = FlatBufferBuilder::new(0);
        assert_eq!(builder.get_buffer(), []);
        let start_offset = builder.start_table();
        assert_eq!(builder.get_buffer(), []);
        builder.add_scalar(4, true, false);
        builder.end_table(start_offset, 1);
        assert_eq!(builder.get_buffer(), [
            6, 0, // vtable bytes
            8, 0, // Length of object, including vtable offset
            7, 0, // Start of bool value
            6, 0, 0, 0, // Offset for start of vtable (int32)
            0, 0, 0, // Padded to 4 bytes
            1, // Bool value
        ]);
    }

    #[test]
    fn test_vtable_with_one_default_bool() {
        let mut builder = FlatBufferBuilder::new(0);
        assert_eq!(builder.get_buffer(), []);
        let start_offset = builder.start_table();
        assert_eq!(builder.get_buffer(), []);
        builder.add_scalar(4, false, false);
        builder.end_table(start_offset, 1);
        assert_eq!(builder.get_buffer(), [
            6, 0, // vtable bytes
            4, 0, // End of object from here
            0, 0, // Entry 1 is zero
            6, 0, 0, 0, // Offset for start of vtable (int32)
        ]);
    }

    #[test]
    fn test_vtable_with_one_int16() {
        let mut builder = FlatBufferBuilder::new(0);
        assert_eq!(builder.get_buffer(), []);
        let start_offset = builder.start_table();
        assert_eq!(builder.get_buffer(), []);
        builder.add_scalar(4, 0x789Ai16, 0);
        builder.end_table(start_offset, 1);
        assert_eq!(builder.get_buffer(), [
            6, 0, // vtable bytes
            8, 0, // End of object from here
            6, 0, // Offset to value
            6, 0, 0, 0, // Offset for start of vtable (int32)
            0, 0, // Padded to 4 bytes
            0x9A, 0x78,
        ]);
    }

    #[test]
    fn test_vtable_with_two_int16() {
        let mut builder = FlatBufferBuilder::new(0);
        let start_offset = builder.start_table();
        builder.add_scalar(4, 0x3456i16, 0);
        builder.add_scalar(6, 0x789Ai16, 0);
        builder.end_table(start_offset, 2);
        assert_eq!(builder.get_buffer(), [
            8, 0, // vtable bytes
            8, 0, // End of object from here
            6, 0, // Offset to value 0
            4, 0, // Offset to value 1
            8, 0, 0, 0, // Offset for start of vtable (int32)
            0x9A, 0x78, // Value 1
            0x56, 0x34, // Value 0
        ]);
    }

    #[test]
    fn test_vtable_with_int16_and_bool() {
        let mut builder = FlatBufferBuilder::new(0);
        let start_offset = builder.start_table();
        builder.add_scalar(4, 0x3456i16, 0);
        builder.add_scalar(6, true, false);
        builder.end_table(start_offset, 2);
        assert_eq!(builder.get_buffer(), [
            8, 0, // vtable bytes
            8, 0, // End of object from here
            6, 0, // Offset to value 0
            5, 0, // Offset to value 1
            8, 0, 0, 0, // Offset for start of vtable (int32)
            0, // Padding
            1, // Value 1
            0x56, 0x34, // Value 0
        ]);
    }

    #[test]
    fn test_vtable_with_empty_vector() {
        let mut builder = FlatBufferBuilder::new(0);
        builder.start_vector(0, mem::size_of::<u8>());
        let vec_offset = Offset::<Vector<u8>>::new(builder.end_vector(0));
        let start_offset = builder.start_table();
        builder.add_offset(4, vec_offset);
        builder.end_table(start_offset, 1);
        assert_eq!(builder.get_buffer(), [
            6, 0, // vtable bytes
            8, 0, // End of object from here
            4, 0, // Offset to vector offset
            6, 0, 0, 0, // Offset for start of vtable (int32)
            4, 0, 0, 0,
            0, 0, 0, 0 // Length of vector (not in struct)
        ]);
    }

    #[test]
    fn test_vtable_with_empty_vector_of_byte_and_some_scalars() {
        let mut builder = FlatBufferBuilder::new(0);
        builder.start_vector(0, mem::size_of::<u8>());
        let vec_offset = Offset::<Vector<u8>>::new(builder.end_vector(0));
        let start_offset = builder.start_table();
        builder.add_scalar(4, 55i16, 0);
        builder.add_offset(6, vec_offset);
        builder.end_table(start_offset, 2);
        assert_eq!(builder.get_buffer(), [
            8, 0, // vtable bytes
            12, 0,
            10, 0, // Offset to value 0
            4, 0, // Offset to vector offset
            8, 0, 0, 0, // Offset for start of vtable (int32)
            8, 0, 0, 0, // Value 1
            0, 0, 55, 0, // Value 0

            0, 0, 0, 0, // Length of vector (not in struct)
        ]);
    }

    #[test]
    fn test_vtable_with_1_int16_and_2vector_of_int16() {
        let mut builder = FlatBufferBuilder::new(0);
        builder.start_vector(2, mem::size_of::<i16>());
        builder.push_scalar(0x1234i16);
        builder.push_scalar(0x5678i16);
        let vec_offset = Offset::<Vector<i16>>::new(builder.end_vector(2));
        let start_offset = builder.start_table();
        builder.add_offset(6, vec_offset);
        builder.add_scalar(4, 55i16, 0);
        builder.end_table(start_offset, 2);
        assert_eq!(builder.get_buffer(), [
            8, 0, // vtable bytes
            12, 0,
            6, 0, // Start of value 0 from end of vtable
            8, 0, // Start of value 1 from end of buffer
            8, 0, 0, 0, // Offset for start of vtable (int32)
            0, 0, // Padding
            55, 0, // Value 0
            4, 0, 0, 0, // Vector position from here
            2, 0, 0, 0, // Length of vector (uint32)
            0x78, 0x56, // Vector value 1
            0x34, 0x12, // Vector value 0
        ]);
    }

    #[test]
    fn test_vtable_with_1_struct_of_1_int8_1_int16_1_int32() {
        let mut builder = FlatBufferBuilder::new(0);
        let start_offset = builder.start_table();
        builder.align(4 + 4 + 4);
        builder.push_scalar(55i8);
        builder.pad(3);
        builder.push_scalar(0x1234i16);
        builder.pad(2);
        builder.push_scalar(0x12345678i32);
        let struct_offset = builder.get_size() as UOffset;
        builder.add_struct_offset(4, struct_offset);
        builder.end_table(start_offset, 1);
        assert_eq!(builder.get_buffer(), [
            6, 0, // vtable bytes
            16, 0, // End of object from here
            4, 0, // Start of struct from here
            6, 0, 0, 0, // Offset for start of vtable (int32)
            0x78, 0x56, 0x34, 0x12, // Value 2
            0, 0, // Padding
            0x34, 0x12, // Value 1
            0, 0, 0, // Padding
            55, // Value 0
        ]);
    }

    #[test]
    fn test_vtable_with_1_vector_of_2_struct_of_2_int8() {
        let mut builder = FlatBufferBuilder::new(0);
        builder.start_vector(2, mem::size_of::<i16>() * 2);
        builder.push_scalar(33i8);
        builder.push_scalar(44i8);
        builder.push_scalar(55i8);
        builder.push_scalar(66i8);
        let vec_offset = Offset::<Vector<i16>>::new(builder.end_vector(2));
        let start_offset = builder.start_table();
        builder.add_offset(4, vec_offset);
        builder.end_table(start_offset, 1);
        assert_eq!(builder.get_buffer(), [
            6, 0, // vtable bytes
            8, 0, // End of object from here
            4, 0, // Offset of vector offset
            6, 0, 0, 0, // Offset for start of vtable (int32)
            4, 0, 0, 0, // Vector start offset

            2, 0, 0, 0, // Vector length
            66, // Vector value 1,1
            55, // Vector value 1,0
            44, // Vector value 0,1
            33, // Vector value 0,0
        ]);
    }

    #[test]
    fn test_table_with_some_elements() {
        let mut builder = FlatBufferBuilder::new(0);
        let start_offset = builder.start_table();
        builder.add_scalar(4, 33i8, 0);
        builder.add_scalar(6, 66i16, 0);
        let offset = Offset::<Table>::new(builder.end_table(start_offset, 2));
        builder.finish(offset);

        assert_eq!(builder.get_buffer(), [
            12, 0, 0, 0, // Root of table, points to vtable offset

            8, 0, // vtable bytes
            8, 0, // End of object from here
            7, 0, // Start of value 0
            4, 0, // Start of value 1

            8, 0, 0, 0, // Offset for start of vtable (int32)

            66, 0, // Value 1
            0, // Padding
            33, // Value 0
        ]);
    }

    #[test]
    fn test_one_unfinished_table_and_one_finished_table() {
        let mut builder = FlatBufferBuilder::new(0);
        let start_offset = builder.start_table();
        builder.add_scalar(4, 33i8, 0);
        builder.add_scalar(6, 44i8, 0);
        let offset = Offset::<Table>::new(builder.end_table(start_offset, 2));
        builder.finish(offset);

        let start_offset = builder.start_table();
        builder.add_scalar(4, 55i8, 0);
        builder.add_scalar(6, 66i8, 0);
        builder.add_scalar(8, 77i8, 0);
        let offset = Offset::<Table>::new(builder.end_table(start_offset, 3));
        builder.finish(offset);

        assert_eq!(builder.get_buffer(), &vec![
            16, 0, 0, 0, // Root of table, points to vtable offset
            0, 0, // Padding

            10, 0, // vtable bytes
            8, 0, // Size of table
            7, 0, // Start of value 0
            6, 0, // Start of value 1
            5, 0, // Start of value 2
            10, 0, 0, 0, // Offset for start of vtable (int32)
            0, // Padding
            77, // Value 2
            66, // Value 1
            55, // Value 0

            12, 0, 0, 0, // Root of table, points to vtable offset

            8, 0, // vtable bytes
            8, 0, // Size of table
            7, 0, // Start of value 0
            6, 0, // Start of value 1
            8, 0, 0, 0, // Offset for start of vtable (int32)
            0, 0, // Padding
            44, // Value 1
            33, // Value 0
        ][..]);
    }

    #[test]
    fn test_a_bunch_of_bools() {
        let mut builder = FlatBufferBuilder::new(0);
        let start_offset = builder.start_table();
        builder.add_scalar(4, true, false);
        builder.add_scalar(6, true, false);
        builder.add_scalar(8, true, false);
        builder.add_scalar(10, true, false);
        builder.add_scalar(12, true, false);
        builder.add_scalar(14, true, false);
        builder.add_scalar(16, true, false);
        builder.add_scalar(18, true, false);
        let offset = Offset::<Table>::new(builder.end_table(start_offset, 8));
        builder.finish(offset);

        assert_eq!(builder.get_buffer(), &vec![
            24, 0, 0, 0, // Root of table, points to vtable offset

            20, 0, // vtable bytes
            12, 0, // Size of table
            11, 0, // Start of value 0
            10, 0, // Start of value 1
            9, 0, // Start of value 2
            8, 0, // Start of value 3
            7, 0, // Start of value 4
            6, 0, // Start of value 5
            5, 0, // Start of value 6
            4, 0, // Start of value 7
            20, 0, 0, 0, // Offset for start of vtable (int32)

            1, // Value 7
            1, // Value 6
            1, // Value 5
            1, // Value 4
            1, // Value 3
            1, // Value 2
            1, // Value 1
            1, // Value 0
        ][..]);
    }

    #[test]
    fn test_three_bools() {
        let mut builder = FlatBufferBuilder::new(0);
        let start_offset = builder.start_table();
        builder.add_scalar(4, true, false);
        builder.add_scalar(6, true, false);
        builder.add_scalar(8, true, false);
        let offset = Offset::<Table>::new(builder.end_table(start_offset, 3));
        builder.finish(offset);

        assert_eq!(builder.get_buffer(), [
            16, 0, 0, 0, // Root of table, points to vtable offset

            0, 0, // Padding

            10, 0, // vtable bytes
            8, 0, // Size of table
            7, 0, // Start of value 0
            6, 0, // Start of value 1
            5, 0, // Start of value 2
            10, 0, 0, 0, // Offset for start of vtable (int32)

            0, // Padding
            1, // Value 2
            1, // Value 1
            1, // Value 0
        ]);
    }

    #[test]
    fn test_some_floats() {
        let mut builder = FlatBufferBuilder::new(0);
        let start_offset = builder.start_table();
        builder.add_scalar(4, 1.0f32, 0.0);
        builder.end_table(start_offset, 1);

        assert_eq!(builder.get_buffer(), [
            6, 0, // vtable bytes
            8, 0, // Size of table
            4, 0, // Start of value 0
            6, 0, 0, 0, // Offset for start of vtable (int32)

            0, 0, 128, 63, // Value 0
        ]);
    }

    #[test]
    fn test_vtable_deduplication() {
        let mut builder = FlatBufferBuilder::new(0);

        let start_offset = builder.start_table();
        builder.add_scalar(4, 0u8, 0);
        builder.add_scalar(6, 11u8, 0);
        builder.add_scalar(8, 22u8, 0);
        builder.add_scalar(10, 33i16, 0);
        builder.end_table(start_offset, 4);

        let start_offset = builder.start_table();
        builder.add_scalar(4, 0u8, 0);
        builder.add_scalar(6, 44u8, 0);
        builder.add_scalar(8, 55u8, 0);
        builder.add_scalar(10, 66i16, 0);
        builder.end_table(start_offset, 4);

        let start_offset = builder.start_table();
        builder.add_scalar(4, 0u8, 0);
        builder.add_scalar(6, 77u8, 0);
        builder.add_scalar(8, 88u8, 0);
        builder.add_scalar(10, 99i16, 0);
        builder.end_table(start_offset, 4);

        //let got = builder.get_buffer();
        assert_eq!(builder.get_buffer(), &vec![
            240, 255, 255, 255, // -12, offset to dedupped vtable
            99, 0,
            88,
            77,
            248, 255, 255, 255, // -8, offset to dedupped vtable
            66, 0,
            55,
            44,
            12, 0,
            8, 0,
            0, 0,
            7, 0,
            6, 0,
            4, 0,
            12, 0, 0, 0,
            33, 0,
            22,
            11,
        ][..]);
    }
}
