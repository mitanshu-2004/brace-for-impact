# What this does not prove

Roughly in order of how much they matter.

## Each robot was trained once

One training run per robot. The confidence intervals in the README describe the testing:
they say how sure I am that a robot with this exact set of weights falls 1.2% of the time.
They say nothing about whether training the same setup again would produce a similar
robot. Reinforcement learning runs vary, sometimes a lot, and the gap between 1.2% and
6.1% could in principle be two runs that happened to land differently.

The crutch result survives this. It compares one robot against itself with the warning
switched off, so there is no second training run to disagree with. The headline
comparison between warned and unwarned does not survive it.

Three to five runs of each, reporting the spread, is the first thing this needs.

## The difficulty ladder ran out

Both robots climbed to the top rung and stayed there, averaging 248.7 N and 242.7 N
against a 250 N ceiling.

I set that ceiling by working out how hard you would need to shove a 13.93 kg object
floating freely in space, which suggested 250 N over 0.3 seconds would be about 3.6 m/s
and unsurvivable. A robot standing on four legs absorbs far more than that, so neither
robot ever reached the point where it was failing half the time.

What that costs is a free second result. Where a robot settles on the ladder is a natural
measure of how good it is, since it is the difficulty at which it survives about half the
time. Neither settled, so the difference between their average rungs, 8.894 and 8.585,
means almost nothing: both are pressed against the same ceiling and the gap only reflects
how often each dipped below it.

The main test is unaffected, because it sets the shove strength directly instead of
reading it off the ladder.

## The two robots did not get identical shoves

Attempts end at different moments for each robot, and a restart immediately picks a new
shove. After the first time they diverge, the two robots are facing the same *kind* of
shoves but not the same *sequence*. So this is a comparison of averages, not a matched
head-to-head.

Doing better would mean recording one schedule of shoves and replaying it against each
robot, with the schedule decoupled from when attempts end.

## The three setups did not get equal numbers of shoves

Between 6,692 and 11,188 shoves each. Falling ends an attempt, restarting brings a fresh
shove, so a robot that falls more collects more shoves in the same wall-clock time.

Since I count falls per shove rather than per attempt, this does not distort the rate
directly. The second-order worry is that a restart also puts the robot back in a clean
standing pose, and a shove arriving just after a restart may be slightly easier to survive
than one arriving mid-stride. That would flatter whichever setup falls most, which is the
warning-switched-off setup, and that one is already losing by a wide margin. So the bias
works against the result I am reporting, not for it.

## Switching the warning off is a test-time trick

It tells you what this trained robot does without its warning. It does not tell you what a
robot trained with an unreliable warning would do. That is the more useful question and
the obvious follow-up: if the warning vanished at random during training, say a quarter of
the time, does the robot keep the ability to catch itself?

## Everything is simulated

No physical robot was involved. Beyond the usual gap between simulation and reality, two
things about the shove are specific to this setup.

The shove is a force applied straight to the robot's body. Nothing actually touches it, so
there is no contact patch, no friction, and nothing deforms. The ball-throwing videos show
the robot against an object that really does collide with it, but no measurement uses
them.

The warning is also perfect. Exactly the right direction, exactly the right strength,
exactly the right timing, every single time. A real robot guessing at an incoming impact
from a camera would be wrong about all three sometimes, and would occasionally warn about
something that never arrives. Given that this robot falls half the time when the warning
merely goes silent, what it does with a *wrong* warning is unknown and probably matters
more than anything measured here.

## "Fell over" is a threshold I chose

A fall is the body tipping past 70 degrees, which is mjlab's default. It is a sensible
line and it is applied identically to all three setups, but the exact percentages would
shift if the line moved. The ordering is much sturdier than the numbers. The tipping-angle
table is reported alongside so you can see the underlying spread rather than only which
side of the line each shove landed on.

## The thrown ball misses sometimes

The ball is aimed by predicting where the robot's body will be, using its speed at the
moment of launch. The warned robot then changes that speed during the flight, slowing and
widening its stance precisely because it was warned, so roughly one throw in three lands
behind it. Measured nearest misses run from 0.16 m to 0.38 m.

Steering the ball mid-flight would fix the hit rate and would also turn it into a guided
missile, which is a worse thing to claim on video. The script checks that at least half
the throws land within 0.20 m of the body and prints every distance, so a genuinely broken
aim still shows up as a failure rather than blending in with this.

The thrown-ball videos are illustration. No number in the results comes from them.
