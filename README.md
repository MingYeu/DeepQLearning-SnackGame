# DeepQLearning Snack Game

It use Apple Silicon GPU to train instead of CPU.

That's how your Snake AI trains itself using Deep Q-Learning!

1. Observing the current state of the game.
2. Taking an action (move left, right, up, down).
3. Receiving a reward (for example, +10 for eating food, -10 for dying).
4. Learning from this experience to make better decisions in the future.

## Decide What Action to Take
final_move = self.get_action(state_old)

It uses the epsilon-greedy policy:
- Random action (exploration) some of the time.
- Action with the highest predicted Q-value (exploitation) most of the time.

## Play the Move and Get Reward
reward, done, score = game.play_step(final_move)

Returns:
- reward: Positive for eating food, negative for dying.
- done: Game over or not.
- score: How many points the snake has.


## Train on Short-Term Memory
self.train_short_memory(state_old, final_move, reward, state_new, done)

## Store the Move in Memory
self.remember(state_old, final_move, reward, state_new, done)

## model.py Training:
self.optimizer.zero_grad()
loss.backward()
self.optimizer.step()

## Save the Model
if score > record:
    agent.model.save()


