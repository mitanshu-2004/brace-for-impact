# Videos

All 42 clips are attached to the
[latest release](https://github.com/mitanshu-2004/go1-brace/releases/latest). They are too
large to keep in the repository itself, but release attachments live on GitHub too, so
there is nothing to sign up for and nowhere else to go.

Every clip is 50 frames per second, encoded at near-lossless quality. The text in the
corner shows which robot it is, how hard the shove is, whether a shove is currently
incoming, and how far the body is tipped at that moment.

## The three robots

- **warned** saw the shove coming during training and can see it now
- **unwarned** never saw shoves coming, during training or now
- **warning_removed** is the warned robot with its warning fed as zeros

## One robot, one shove strength

`warned_35N.mp4` through `warned_250N.mp4`, and the same for `unwarned_` and
`warning_removed_`. Twenty-one clips, 18 seconds each, three to six shoves per clip.

Start with `warned_250N.mp4` and `unwarned_250N.mp4` side by side. That is where the
difference is largest.

## Comparisons

`compare_35N.mp4` through `compare_250N.mp4` put all three robots in one frame at the same
shove strength.

These are three separate runs, not the same shoves replayed three times. Attempts end at
different moments, which changes when the next shove arrives, so they cannot be
synchronised without faking it. Treat them as an illustration of the difference, not as
the measurement.

`warned_all_forces.mp4`, `unwarned_all_forces.mp4` and `warning_removed_all_forces.mp4`
show one robot at all seven shove strengths at once, gentlest top-left to hardest
bottom-right.

## Learning from scratch

`warned_learning.mp4` and `unwarned_learning.mp4` step through all 81 saved checkpoints in
order, two seconds each, at a constant 250 N shove. The counter in the corner is the
training round. Two minutes forty-two seconds from a robot that can barely stand to the
finished one.

## A real ball

`warned_ball.mp4` and `unwarned_ball.mp4` throw an actual 5 kg ball at the robot instead
of applying a force to it. Same trained robot, unchanged; only the way the shove is
delivered is different. `_short` versions cut out the walking between throws.

No number in the results comes from these. The measurements all use a directly applied
force, because that gives an exact strength to put on the chart. The ball is there because
"a force was applied to the body" is not something you can watch.

About one throw in three misses, since the ball is aimed at where the robot is going to be
and the warned robot slows down when it sees the throw coming.

## Close-ups

`warned_250N_orbit.mp4` circles the robot while it takes hits. `warned_250N_closeup.mp4`
is a low, near view of the same thing.

## A note on the word "IMPACT"

The corner text switches to IMPACT when the shove is due to land, which is a countdown
reaching zero rather than a contact sensor. In the force clips the force really is being
applied at that moment. In the ball clips no force is applied at all, so IMPACT marks when
the ball was *supposed* to arrive, which is not always when or whether it hit.

## The warning arrow

A red arrow closes in on the robot before each shove, showing where it is coming from, and
turns magenta at the moment of impact.

The `warning_removed` clips still draw the arrow. You can see the shove coming; the robot
cannot. That is the whole point of that condition, and the corner text says so.
