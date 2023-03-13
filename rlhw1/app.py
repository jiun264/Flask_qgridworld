from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Define the Q-learning algorithm function


def q_learning(grid, start, end, obstacles, gamma=0.7, alpha=0.5, epsilon=0.1, num_episodes=100):
    # Get the grid size
    print('Q-learning is running')
    n = len(grid)

    # Initialize the Q-values to zeros
    q_values = np.zeros((n, n, 4))

    # Define the actions: up, down, left, right
    actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # Define the reward function
    def get_reward(state):
        if state == end:
            return 100
        elif state in obstacles:
            return -100
        else:
            return -1

    # Define the epsilon-greedy policy
    def epsilon_greedy(state, epsilon):
        if np.random.uniform() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_values[state[0]][state[1], :])
        return action

    # Run the Q-learning algorithm for the specified number of episodes
    for episode in range(num_episodes):
        # Reset the current state to the start state
        state = start
        # Loop until the goal is reached or an obstacle is hit
        while state != end and state not in obstacles:
            # Choose an action based on the epsilon-greedy policy
            action = epsilon_greedy(state, epsilon)
            # Update the Q-value for the current state-action pair
            next_state = (state[0] + actions[action][0],
                          state[1] + actions[action][1])
            if next_state[0] < 0 or next_state[0] >= n or next_state[1] < 0 or next_state[1] >= n:
                  # Next state is out of bounds, so stay in the current state
                next_state = state
            # if next_state[0] < 0 or next_state[0] >= grid.shape[0] or next_state[1] < 0 or next_state[1] >= grid.shape[1]:
            #     # Next state is out of bounds, so ignore this move
            #     continue
            reward = get_reward(next_state)
            q_values[state[0], state[1], action] += alpha * (reward + gamma * np.max(
                q_values[next_state[0], next_state[1]]) - q_values[state[0], state[1], action])
            # Move to the next state
            state = next_state
    return q_values


# Define a function to get the shortest path based on the Q-values
def get_path(q_values, start, end):
    # Define the actions: up, down, left, right
    actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # Initialize the current state to the start state
    state = start
    # Initialize the path with the start state
    path = [state]

    # Loop until the goal is reached or an obstacle is hit
    while state != end:
        # Choose the action with the highest Q-value for the current state
        action = np.argmax(q_values[state[0], state[1]])

        # Move to the next state based on the chosen action
        state = (state[0] + actions[action][0], state[1] + actions[action][1])
        # Add the new state to the path
        path.append(state)

    return path


# Define the Flask routes

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Square generator page


@app.route('/square', methods=['GET', 'POST'])
def square():
    if request.method == 'POST':
        n = int(request.form['n'])
        return render_template('square.html', n=n)
    else:
        return render_template('square.html')

# Q-learning solver API


@app.route('/solve', methods=['POST'])
def solve():
    # Get the input parameters from the POST request
    grid = np.array(request.json['grid'])
    # start = tuple(request.json['start'])
    start = tuple(map(int, request.json['start'].split(',')))
    # end = tuple(request.json['end'])
    end = tuple(map(int, request.json['end'].split(',')))

    # obstacles = [tuple(obstacle) for obstacle in request.json['obstacles']]
    obstacles = [tuple(map(int, obstacle.split(',')))
                 for obstacle in request.json['obstacles']]

    # Run the Q-learning algorithm
    q_values = q_learning(grid, start, end, obstacles)
    path = get_path(q_values, start, end)
    # Return the Q-values as a JSON response
    # return jsonify(q_values.tolist())
    return jsonify(path=path)

if __name__ == '__main__':
    app.run(debug=True)
