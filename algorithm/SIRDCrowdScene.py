import collections
import random
from threading import Lock


class SIRDCrowdScene:
    def __init__(self, initial_num, area_size, velocity_rate, infection_radius, infection_rate, recovery_rate, mortality_rate, initial_infected_num):

        self.initial_num = initial_num
        self.area_size = area_size
        self.velocity_rate = velocity_rate
        self.points = []
        self.infection_radius = infection_radius
        # 网格尺寸为感染半径的2倍，恰好确保邻近检测覆盖
        self.grid_size = self.infection_radius * 2
        # 存储网格坐标对应的点
        self.grid = collections.defaultdict(list)
        self.grid_lock = Lock()

        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.mortality_rate = mortality_rate

        for _ in range(self.initial_num):
            p = {
                "s": None,
                "x": random.uniform(0, self.area_size),
                "y": random.uniform(0, self.area_size),
                "vx": random.uniform(-1, 1),
                "vy": random.uniform(-1, 1),
            }
            self.points.append(p)
            grid_pos = (int(p['x'] // self.grid_size), int(p['y'] // self.grid_size))
            self.grid[grid_pos].append(p)

        # 初始化感染者
        initial_infected_num = min(initial_infected_num, self.initial_num)
        infected_indices = random.sample(range(self.initial_num), initial_infected_num)
        for idx in infected_indices:
            self.points[idx]['s'] = 'I'

    def update_position(self):
        with self.grid_lock:
            for p in self.points:
                # 记录旧网格
                old_x, old_y = p['x'], p['y']
                old_grid = (int(old_x // self.grid_size), int(old_y // self.grid_size))

                # 更新位置
                p['x'] += p['vx']
                p['y'] += p['vy']

                # 边界反弹
                if p['x'] < 0 or self.area_size:
                    p['vx'] *= -1
                    p['x'] = max(0, min(p['x'], self.area_size))
                if p['y'] < 0 or self.area_size:
                    p['vy'] *= -1
                    p['y'] = max(0, min(p['y'], self.area_size))

                # 更新网格
                new_grid = (int(p['x'] // self.grid_size), int(p['y'] // self.grid_size))
                if new_grid != old_grid:
                    self.grid[old_grid].remove(p)
                    self.grid[new_grid].append(p)
                    
    def update_states(self):
        with self.grid_lock:
            for p in self.points:
                if p['s'] == 'I':
                    grid_x, grid_y = int(p['x'] // self.grid_size), int(p['y'] // self.grid_size)
                    # 检查3*3网格
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            neighbor_grid = (grid_x + dx, grid_y + dy)
                            for neighbor in self.grid.get(neighbor_grid, []):
                                if neighbor['s'] == 'S' and self.calculate_distance(p, neighbor) < self.infection_radius:
                                    if random.random() < self.infection_rate:
                                        neighbor['s'] = 'I'

            for p in self.points:
                if p['s'] == 'I':
                    prob = random.random()
                    if prob < self.recovery_rate:
                        p['s'] = 'R'
                    elif prob < self.recovery_rate + self.mortality_rate:
                        p['s'] = 'D'

    def calculate_distance(self, p1, p2):
        return ((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)**0.5

    def update(self):
        self.update_position()
        self.update_states()

