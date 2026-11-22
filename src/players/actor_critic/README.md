# Pure Actor-Critic (exercise skeleton)

`players.actor_critic` is an agent that plays from a policy network alone. There is no tree search and no forward-model simulation. Each `act()` call is a single network call:

```
pi(. | s), V(s) = network(s, legal actions)
a ~ pi                          (argmax when evaluating)
```

The work is split between two places:

- **Java** (this package) **acts and records**: it plays games with the current policy and writes trajectories to disk.
- **Python** (which you write) **learns**: it computes returns and advantages, updates the actor and the critic, and exports the new policy.

Every function is an empty stub marked `TODO`. Non-void stubs throw `UnsupportedOperationException`, so you'll know straight away if you call one before writing it. For the MCTS-with-priors version, see [`players.actor_critic_mcts`](../actor_critic_mcts/README.md).

## Layout

| File | Role |
|------|------|
| [ActorCriticPlayer.java](ActorCriticPlayer.java) | Agent: `act`, `chooseAction`, `buildTransition`, `startRecording` and `result` |
| [ActorCriticParams.java](ActorCriticParams.java) | Sampling or argmax, temperature, recording, `GAMMA` and reward-shaping weight |
| [network/ActorCriticNetwork.java](network/ActorCriticNetwork.java) | Interface: `evaluate(gs, actions, playerID) -> PolicyValue` |
| [network/PolicyValue.java](network/PolicyValue.java) | `probs[]` (pi over the legal actions) + `value` (V(s)) |
| [network/ExternalNetwork.java](network/ExternalNetwork.java) | Bridge to the model trained in Python (ONNX, socket or hand-written forward pass) |
| [encoding/StateEncoder.java](encoding/StateEncoder.java) | `GameState` → feature vector |
| [encoding/ActionEncoder.java](encoding/ActionEncoder.java) | `Action` → fixed index (scheme A) or feature vector (scheme B) |
| [policy/PolicySampler.java](policy/PolicySampler.java) | Masked softmax, temperature, sampling, argmax and entropy |
| [reward/RewardFunction.java](reward/RewardFunction.java) | Step reward (optional heuristic shaping) and terminal reward (win or loss) |
| [records/Transition.java](records/Transition.java) | (s, legal actions, a, pi_old(a\|s), V(s), r, done) |
| [records/Trajectory.java](records/Trajectory.java) | One player's transitions in one game; rewards are filled in one step late |
| [records/RecordWriter.java](records/RecordWriter.java) | Writes trajectories as JSON Lines |
| [selfplay/SelfPlayConfig.java](selfplay/SelfPlayConfig.java) | Number of games, seeds, tribes and paths |
| [selfplay/SelfPlay.java](selfplay/SelfPlay.java) | `main`: plays games headless with the current policy and writes the trajectories |

## Data flow

```
SelfPlay.playGame
  ├─ Game.init → startRecording (after init, once playerID is set)
  └─ Game.run(null,null)
       ├─ ActorCriticPlayer.act(s_t)
       │    ├─ trajectory.setLastReward( RewardFunction.stepReward(s_{t-1}, s_t) )
       │    ├─ network.evaluate(s_t) → pi, V
       │    ├─ a_t = chooseAction(pi)
       │    └─ trajectory.add(Transition{s_t, a_t, pi(a_t), V(s_t)})
       └─ Agent.result(final) → trajectory.finish( RewardFunction.terminalReward(...) )
  └─ RecordWriter.writeAll → data/ac/iter_XXX/*.jsonl → Python trainer → policy_{k+1}
```

Here `s_t` is the state at **this player's** t-th decision. The opponents' moves between `s_t` and `s_{t+1}` count as part of the environment.

## What the Python trainer has to do (not included)

1. Group the transitions by (gameId, playerID) and keep them in order.
2. Compute the targets from `reward`, `value` and `done`: Monte-Carlo returns, n-step returns, or GAE(γ, λ). Use the same `GAMMA` as `ActorCriticParams`.
3. Advantage A_t = target_t − V(s_t), usually normalised per batch.
4. Losses: actor = −log pi(a_t | s_t) · A_t (minus an entropy bonus); critic = (V(s_t) − target_t)². For PPO, use the ratio pi_new / `actionProb`.
5. Export the updated model for `ExternalNetwork`, then collect fresh data. Old data came from an older policy, so don't reuse it without a correction (such as the PPO ratio).

## Suggested order

1. `policy/PolicySampler.java`: pure functions, so you can unit-test them on their own.
2. `encoding/*`: choose scheme A or B for actions. The encoding defines the network's input, so do this before writing the model.
3. `reward/RewardFunction.java`: start with terminal reward plus a small shaping term.
4. `records/*`, then `ActorCriticPlayer`.
5. `network/ExternalNetwork.java`. For iteration 0, export a randomly initialised model so the first data really comes from the policy you train.
6. `selfplay/*`, then the Python trainer. Afterwards, evaluate the policy with `SAMPLE_ACTIONS = false` against `RandomAgent`, then against MCTS.

## Running

```bash
javac -cp lib/json.jar -d out/classes $(find src -name "*.java")
java -cp out/classes:lib/json.jar players.actor_critic.selfplay.SelfPlay
```

The agent isn't registered in `Run.PlayerType` / `Run.getAgent()`. Add an entry there when you want to use it from `play.json` or `tournament.json`.
