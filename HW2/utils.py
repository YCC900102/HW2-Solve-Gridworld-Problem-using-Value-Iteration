import numpy as np

ACTIONS = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

ACTION_SYMBOL = {
    'U': '↑',
    'D': '↓',
    'L': '←',
    'R': '→'
}

def in_grid(x, y, size):
    return 0 <= x < size and 0 <= y < size

def evaluate_policy(grid_size, start, end, obstacles, policy, gamma=0.9, theta=1e-4):
    V = np.zeros((grid_size, grid_size))
    obstacle_set = set(map(tuple, obstacles))
    end_pos = tuple(end)

    def is_terminal(i, j):
        return (i, j) == end_pos

    def is_obstacle(i, j):
        return (i, j) in obstacle_set

    while True:
        delta = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if is_terminal(i, j) or is_obstacle(i, j):
                    continue
                action = policy[i][j]
                dx, dy = ACTIONS[action]
                ni, nj = i + dx, j + dy
                if not in_grid(ni, nj, grid_size) or is_obstacle(ni, nj):
                    ni, nj = i, j
                reward = 0 if (ni, nj) == end_pos else -1
                new_v = reward + gamma * V[ni, nj]
                delta = max(delta, abs(V[i, j] - new_v))
                V[i, j] = new_v
        if delta < theta:
            break
    return V

def policy_iteration(grid_size, start, end, obstacles, gamma=0.9, theta=1e-4):
    policy = [[np.random.choice(list(ACTIONS.keys())) for _ in range(grid_size)] for _ in range(grid_size)]
    obstacle_set = set(map(tuple, obstacles))
    end_pos = tuple(end)

    def is_terminal(i, j):
        return (i, j) == end_pos

    def is_obstacle(i, j):
        return (i, j) in obstacle_set

    while True:
        V = evaluate_policy(grid_size, start, end, obstacles, policy, gamma, theta)
        policy_stable = True
        for i in range(grid_size):
            for j in range(grid_size):
                if is_terminal(i, j) or is_obstacle(i, j):
                    continue
                old_action = policy[i][j]
                best_action = None
                best_value = float('-inf')
                for action, (dx, dy) in ACTIONS.items():
                    ni, nj = i + dx, j + dy
                    if not in_grid(ni, nj, grid_size) or is_obstacle(ni, nj):
                        ni, nj = i, j
                    reward = 0 if (ni, nj) == end_pos else -1
                    value = reward + gamma * V[ni, nj]
                    if value > best_value:
                        best_value = value
                        best_action = action
                policy[i][j] = best_action
                if best_action != old_action:
                    policy_stable = False
        if policy_stable:
            break

    # 產出對應的 symbol 與 value matrix
    policy_symbol = [['' for _ in range(grid_size)] for _ in range(grid_size)]
    for i in range(grid_size):
        for j in range(grid_size):
            if is_terminal(i, j):
                policy_symbol[i][j] = 'G'
            elif is_obstacle(i, j):
                policy_symbol[i][j] = 'X'
            else:
                policy_symbol[i][j] = ACTION_SYMBOL[policy[i][j]]

    return V, policy_symbol
