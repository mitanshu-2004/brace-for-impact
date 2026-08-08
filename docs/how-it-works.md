# How it works

## The task

The robot is a Unitree Go1, a 13.93 kg four-legged robot. It walks on flat ground
following speed commands, using mjlab's existing walking task. The shoving is added on
top rather than replacing it.

Keeping the walking requirement matters. If the robot only had to stay upright, the
easiest answer is to crouch, spread out, and never move. Since it also has to hit a
target walking speed, that answer stops working. Two safety checks back this up, but the
walking reward does most of the work: a robot lying on the ground is not walking
anywhere.

### The built-in push is removed

mjlab's walking task already shoves the robot: every 1 to 3 seconds it instantly changes
the robot's velocity by up to 0.5 m/s. Its own description calls this "instantaneous,
mass-independent ... ignoring inertia and contact dynamics".

That is switched off here. It is not a force, so it cannot be aimed at a spot on the body,
scaled with difficulty, or announced ahead of time. Leaving it on would also make any
claim like "the normal robot falls over when pushed" false, since the normal robot is
already trained against those pushes.

## The shove

Each robot runs its own loop:

```
pick a shove -> count down (robot can see it coming) -> apply the force
             -> hold briefly -> stop -> wait -> pick the next one
```

What the robot senses is four numbers:

```
[cos(direction), sin(direction), how hard, how long until it lands]
```

The direction is relative to the robot's own heading, not the world, because what matters
is whether the shove is coming from its left or its front. All four numbers are zero when
nothing is coming, so "nothing is coming" is unmistakable rather than something the robot
has to learn to recognise.

The last two numbers are divided by fixed constants, deliberately not by the current
range. Testing pins the shove strength to one value at a time, so dividing by the range
maximum would make the robot read "as hard as it gets" at every strength.

### Where the shove lands

The force is applied to the robot's body, 10 cm above its centre. That offset is what
makes this interesting. A force through the centre slides the robot sideways. A force
above the centre also rotates it, so it tips. Recovering from a tip means moving the feet,
which is the behaviour being studied.

Forces in MuJoCo stay applied until something removes them, so the code clears the force
when the shove ends and again whenever an attempt restarts. Without the second clear, a
shove still in progress when a robot falls would carry into its next life.

### How hard

For a force `F` held for time `dt` on a mass `m`, the speed change is `F * dt / m`. The
ladder runs from 30 N to 250 N, held between 0.1 and 0.3 seconds. On a free-floating
13.93 kg object that spans roughly 0.7 m/s to 3.6 m/s.

"Free-floating" is why the ceiling turned out too low. A robot with four feet on the
ground absorbs much more than that calculation suggests, and both robots ended up pinned
at the top of the ladder. See [limitations.md](limitations.md).

### The difficulty ladder

Difficulty is tracked separately for each of the 4096 robots training in parallel. Finish
an attempt and you move up a rung; fall and you move down. Every robot therefore spends
its time near its own limit instead of all of them sharing one difficulty that is too easy
for some and impossible for others.

Ten rungs, spread evenly across the force range.

## When a robot counts as fallen

An attempt (an "episode", in the usual terminology) ends one of three ways, all from
mjlab:

- 1000 steps elapse
- the body tips past 70 degrees, which is the definition of "fell" used everywhere in the
  results
- the body drops below 0.12 m, which catches the crouch-and-hide strategy

That 0.12 m is measured, not guessed. A robot doing nothing at all sags to 0.151 m, so a
0.15 m limit would end almost every attempt in early training before anything could be
learned. A properly collapsed robot sits around 0.08 m. At 0.12 m this check triggers on
0.042 attempts per run versus 1.21 for tipping over, so it acts as a backstop rather than
the thing driving behaviour.

## Rewards

None were added. mjlab already rewards staying upright and hitting the target speed.
Writing new reward functions would have introduced differences between this task and the
standard one that have nothing to do with the question being asked.

## The three setups

| setup | can the robot see the warning | can the value estimator see it | trained separately |
|---|---|---|---|
| warned | yes | yes | yes |
| not warned | no | yes | yes |
| warning switched off | no, zeroed at test time | yes | no, reuses the warned robot |

The value estimator sees the warning in both trained setups. It is only used during
training, to judge how good a situation is, and is thrown away afterwards. Giving it to
both means the two robots are judged equally well and the only real difference is what the
action-picking part is allowed to use. Otherwise the unwarned robot would be penalised
twice: worse information *and* a worse teacher.

The third setup exists because two results are easy to confuse. If the two robots had come
out equal, that could mean the warning is useless, or it could mean the warned robot never
figured out how to use it. Switching the warning off and seeing what happens separates
them.

It is done by replacing the function that supplies those four numbers with one that
returns zeros. The robot's input is the same size, so the saved weights still load, there
is no risk of shifting the other inputs by four places, and zeros are exactly what the
robot already sees between shoves, so it is not being fed nonsense.

## Measuring

`eval.py` freezes a robot, fixes the shove strength, and writes one line per shove:

```
force_n, bearing_rad, peak_tilt_deg, displacement_m, watched_steps, fell, robot
```

Every number in the README comes from these lines, so a different way of summarising them
does not need another test run.

The strength has to be fixed for this to mean anything. During training the difficulty
ladder ties shove strength to how good the robot currently is, so a fall-rate-versus-
strength curve taken from training would show every robot being hit exactly as hard as it
can just barely survive. That describes the ladder, not the robot.

Falls are counted per shove, not per attempt, because one attempt can contain several
shoves and "survived four" should not count the same as "survived one".

## Where it ran

Training: a free Kaggle T4. 4096 robots in parallel, about 2.4 hours each, roughly 49,000
simulated steps per second. The two robots are independent and trained at the same time.

Video: a Lightning T4, because MuJoCo needs proper NVIDIA graphics drivers that the Kaggle
image does not have. A 900-frame 1080p clip takes about 80 seconds including startup.
