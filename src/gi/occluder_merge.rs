//! Merging rows of touching occluders into one, before upload.
//!
//! The SDF pass measures every texel against every occluder, so its cost is
//! texels times occluders. A tilemap gives each tile its own box, and a row of
//! tiles is a row of equal boxes side by side: one box as long as the row has
//! the same outside, so the same SDF wherever light travels. Only the inside
//! changes: a row of tiles reads zero along every seam between two tiles, the
//! merged box only at its own edge.

use core::hash::BuildHasher;

use bevy::platform::hash::FixedHasher;
use bevy::prelude::*;

/// One occluder as the SDF pass sees it: a box of half size `h_extent`,
/// turned by `rotation`, around `center`, its corners rounded by `radius`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OccluderBox
{
    pub center:   Vec2,
    pub rotation: Quat,
    pub h_extent: Vec2,
    pub radius:   f32,
}

/// How near two lengths count as equal, as a fraction of the box's size:
/// tiles placed from a grid agree to float rounding, far inside this.
const TOLERANCE: f32 = 1e-3;

/// What boxes share to be in one row, quantized by [`TOLERANCE`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RowKey
{
    rotation: IVec4,
    h_extent: IVec2,
    across:   i64,
}

/// The last frame's boxes and what they merged to, so a frame where no
/// occluder moved, came or went reuses them instead of sorting again.
#[derive(Default)]
pub(crate) struct RowMerger
{
    input:  Vec<OccluderBox>,
    output: Vec<OccluderBox>,
}

impl RowMerger
{
    pub(crate) fn merge(&mut self, boxes: Vec<OccluderBox>) -> &[OccluderBox]
    {
        if boxes != self.input {
            self.output = merge_rows(&boxes);
            self.input = boxes;
        }
        &self.output
    }
}

/// Merges each run of sharp boxes that share rotation and size, sit on one
/// line along their own x axis and touch end to end, into one box. Anything
/// else passes through as it was: two rounded boxes end to end have rounded
/// corners at their seam that one longer box has not. The order of the
/// result is not the input's.
pub(crate) fn merge_rows(boxes: &[OccluderBox]) -> Vec<OccluderBox>
{
    // A box's row is what it shares with the boxes it can merge with: its
    // rotation, its size and its offset across the row. Sorted, each row's
    // boxes lie next to each other in order along it. It runs in any frame
    // where an occluder moved, so the sort moves small entries (the row's
    // hash, the place along it, the box's index) and the scan checks the row
    // itself: two rows that hash alike only split a run, never join.
    let rows: Vec<RowKey> = boxes
        .iter()
        .map(|b| {
            let across = b.rotation * Vec3::Y;
            let unit = b.h_extent.max_element().max(f32::EPSILON) * TOLERANCE;
            RowKey {
                rotation: (Vec4::from(b.rotation) / TOLERANCE).round().as_ivec4(),
                h_extent: (b.h_extent / unit).round().as_ivec2(),
                across:   (b.center.dot(across.xy()) / unit).round() as i64,
            }
        })
        .collect();
    let mut order: Vec<(u64, f32, u32)> = boxes
        .iter()
        .zip(&rows)
        .enumerate()
        .map(|(i, (b, row))| {
            (
                FixedHasher.hash_one(row),
                b.center.dot((b.rotation * Vec3::X).xy()),
                i as u32,
            )
        })
        .collect();
    order.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));

    let mut merged = Vec::new();
    let mut run_start = 0;
    for i in 1..=order.len() {
        let touches =
            i < order.len() && rows[order[i].2 as usize] == rows[order[i - 1].2 as usize] && {
                let b = boxes[order[i].2 as usize];
                let step = order[i].1 - order[i - 1].1;
                b.radius == 0.0
                    && boxes[order[i - 1].2 as usize].radius == 0.0
                    && (step - 2.0 * b.h_extent.x).abs() <= b.h_extent.max_element() * TOLERANCE
            };
        if touches {
            continue;
        }
        let (first, last) = (
            boxes[order[run_start].2 as usize],
            boxes[order[i - 1].2 as usize],
        );
        let reach = (order[i - 1].1 - order[run_start].1) / 2.0;
        merged.push(OccluderBox {
            center:   (first.center + last.center) / 2.0,
            rotation: first.rotation,
            h_extent: Vec2::new(first.h_extent.x + reach, first.h_extent.y),
            radius:   first.radius,
        });
        run_start = i;
    }
    merged
}

#[cfg(test)]
mod tests
{
    use super::*;

    fn tile(x: f32, y: f32) -> OccluderBox
    {
        OccluderBox {
            center:   Vec2::new(x, y),
            rotation: Quat::IDENTITY,
            h_extent: Vec2::splat(0.25),
            radius:   0.0,
        }
    }

    #[test]
    fn a_row_of_touching_tiles_becomes_one_box()
    {
        let row: Vec<_> = (0..4).map(|i| tile(i as f32 * 0.5, 1.0)).collect();
        let merged = merge_rows(&row);
        assert_eq!(merged.len(), 1);
        assert!(merged[0].center.abs_diff_eq(Vec2::new(0.75, 1.0), 1e-6));
        assert!(merged[0].h_extent.abs_diff_eq(Vec2::new(1.0, 0.25), 1e-6));
    }

    #[test]
    fn a_gap_splits_the_row_and_rows_stay_apart()
    {
        let boxes = [
            tile(0.0, 0.0),
            tile(0.5, 0.0),
            tile(1.5, 0.0),
            tile(0.0, 0.5),
        ];
        let mut merged = merge_rows(&boxes);
        merged.sort_by(|a, b| {
            (a.center.y, a.center.x)
                .partial_cmp(&(b.center.y, b.center.x))
                .unwrap()
        });
        assert_eq!(merged.len(), 3);
        assert!(merged[0].h_extent.abs_diff_eq(Vec2::new(0.5, 0.25), 1e-6));
        assert!(merged[1].center.abs_diff_eq(Vec2::new(1.5, 0.0), 1e-6));
        assert!(merged[2].center.abs_diff_eq(Vec2::new(0.0, 0.5), 1e-6));
    }

    #[test]
    fn a_turned_row_merges_along_its_own_axis()
    {
        let rotation = Quat::from_rotation_z(0.3);
        let along = (rotation * Vec3::X).xy();
        let row: Vec<_> = (0..3)
            .map(|i| OccluderBox {
                center: Vec2::new(5.0, -2.0) + along * (i as f32 * 0.5),
                rotation,
                h_extent: Vec2::splat(0.25),
                radius: 0.0,
            })
            .collect();
        let merged = merge_rows(&row);
        assert_eq!(merged.len(), 1);
        assert!(merged[0]
            .center
            .abs_diff_eq(Vec2::new(5.0, -2.0) + along * 0.5, 1e-5));
        assert!(merged[0].h_extent.abs_diff_eq(Vec2::new(0.75, 0.25), 1e-5));
    }

    #[test]
    fn different_sizes_or_turns_do_not_merge()
    {
        let mut big = tile(0.5, 0.0);
        big.h_extent = Vec2::splat(0.3);
        let mut turned = tile(1.0, 0.0);
        turned.rotation = Quat::from_rotation_z(0.1);
        assert_eq!(merge_rows(&[tile(0.0, 0.0), big, turned]).len(), 3);
    }

    #[test]
    fn rounded_boxes_do_not_merge()
    {
        let mut round = tile(0.5, 0.0);
        round.radius = 0.1;
        let mut merged = merge_rows(&[tile(0.0, 0.0), round, tile(1.0, 0.0)]);
        merged.sort_by(|a, b| a.center.x.partial_cmp(&b.center.x).unwrap());
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[1].radius, 0.1);
    }

    #[test]
    fn the_merger_follows_a_change_and_keeps_the_rest()
    {
        let mut merger = RowMerger::default();
        let row: Vec<_> = (0..3).map(|i| tile(i as f32 * 0.5, 0.0)).collect();
        assert_eq!(merger.merge(row.clone()).len(), 1);
        assert_eq!(merger.merge(row.clone()).len(), 1);
        let mut broken = row;
        broken[1].center.y += 1.0;
        assert_eq!(merger.merge(broken).len(), 3);
    }
}
