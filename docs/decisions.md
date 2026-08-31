# Open decisions

## 1. Should `stations` keep drawing nothing?

`stations` emits width staircases and no outline sketch, so 46% of parts produce **no
drawing at all** - region IoU 0.000, not merely poor. Routing those parts to an outline
instead was measured on the same 134 parts:

| | sketch IoU | solid IoU |
|---|---|---|
| master, routed, keeps `stations` | 0.269 | **0.525** |
| always outline (revolve still used for round parts) | **0.557** | 0.443 |

**+0.288 sketch, -0.082 solid.**

`tests/test_regression.py` currently asserts all four modes stay reachable and pins four
parts to `stations`, guarding a solid-IoU floor. The change was made, measured, and
reverted to leave that guard authoritative.

A reconciliation was attempted and failed: building the stations solid from the traced
outline intersected with the side elevation scores 0.218, worse than either alone, for
the same registration reason that sinks every multi-view intersection here.

So it is a straight choice. The recommendation is **always-outline**: the product is the
sketch, a parametric sketch can be edited to fix a wrong thickness while a staircase
solid with no sketch cannot, and the solid being given up (0.44 against 0.53) is fairly
wrong either way - the real solid fix is carving at 0.79.

## 2. Shoot one part on the ChArUco target

Carving is validated at **0.791** volumetric IoU with sub-millimetre extent error, against
the photo pipeline's 0.43, and it *measures* depth rather than predicting a 10x band. It
cannot be exercised on PrintCAD at all because those photos have no target.

Sixteen or more photos of a single part on the board, **including grazing angles near
the board plane** - that is what resolves thickness. One part is enough to confirm the
algorithm survives real segmentation and real pose estimation.

Note from the capture measurement: the target's value is knowing the pose and the scale,
**not** undoing perspective, which costs almost nothing. Even without the target,
holding the camera square to the face is worth about +0.13.
