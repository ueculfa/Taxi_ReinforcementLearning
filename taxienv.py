import numpy as np
from gym import Env, spaces


class Taxi6x6Env(Env):
    """
    6x6 Taxi ortamı
    - 6x6 grid
    - 4 nokta: R, G, Y, B (köşeler)
    - Yolcu 4 noktadan birinde ya da takside (4)
    - 3 adet kapalı hücre (duvar) -> taksi bu hücreye giremez
    - Aksiyonlar: 0:Güney, 1:Kuzey, 2:Doğu, 3:Batı, 4:Pickup, 5:Dropoff
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.num_rows = 6
        self.num_cols = 6

        # Yolcu / hedef lokasyonları (R, G, Y, B)
        # Köşeler: (0,0) R, (0,5) G, (5,0) Y, (5,5) B
        self.locs = [(0, 0), (0, 5), (5, 0), (5, 5)]
        self.num_pass_locs = len(self.locs) + 1  # +1 = takside
        self.num_dest_locs = len(self.locs)

        self.num_states = (
            self.num_rows * self.num_cols * self.num_pass_locs * self.num_dest_locs
        )
        self.num_actions = 6

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)

        self.render_mode = render_mode

        # DUVARLAR = KAPALI HÜCRELER
        # Bu hücrelere taksi giremez, sadece görsel olarak gri kutu olarak çizilecek
        # İstediğin gibi değiştirebilirsin, şimdilik örnek:
        self.blocked_cells = {
            (1, 2),
            (3, 1),
            (2, 4),
        }

        # Bu hücrelerin köşe lokasyonlarıyla çakışmamasına dikkat et
        for bc in self.blocked_cells:
            if bc in self.locs:
                raise ValueError("Blocked cell, locs ile çakışıyor!")

        # Geçiş olasılıkları ve başlangıç dağılımı
        self.initial_state_distrib = np.zeros(self.num_states, dtype=float)
        self.P = {
            state: {action: [] for action in range(self.num_actions)}
            for state in range(self.num_states)
        }

        max_row = self.num_rows - 1
        max_col = self.num_cols - 1

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # Taksi bu hücrede olamaz → bu state'leri tamamen yok say
                if (row, col) in self.blocked_cells:
                    continue

                for pass_idx in range(self.num_pass_locs):  # 0..3 + taksi(4)
                    for dest_idx in range(self.num_dest_locs):  # 0..3
                        state = self.encode(row, col, pass_idx, dest_idx)

                        # Başlangıç dağılımı: yolcu hedefte başlamasın
                        if pass_idx < 4 and pass_idx != dest_idx:
                            self.initial_state_distrib[state] += 1

                        for action in range(self.num_actions):
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = -1
                            terminated = False
                            taxi_loc = (row, col)

                            # Hareketler
                            if action == 0:  # south
                                cand_row = min(row + 1, max_row)
                                cand_col = col
                                if (cand_row, cand_col) not in self.blocked_cells:
                                    new_row, new_col = cand_row, cand_col
                            elif action == 1:  # north
                                cand_row = max(row - 1, 0)
                                cand_col = col
                                if (cand_row, cand_col) not in self.blocked_cells:
                                    new_row, new_col = cand_row, cand_col
                            elif action == 2:  # east
                                cand_row = row
                                cand_col = min(col + 1, max_col)
                                if (cand_row, cand_col) not in self.blocked_cells:
                                    new_row, new_col = cand_row, cand_col
                            elif action == 3:  # west
                                cand_row = row
                                cand_col = max(col - 1, 0)
                                if (cand_row, cand_col) not in self.blocked_cells:
                                    new_row, new_col = cand_row, cand_col

                            # Pickup
                            elif action == 4:
                                if pass_idx < 4 and taxi_loc == self.locs[pass_idx]:
                                    new_pass_idx = 4  # takside
                                else:
                                    reward = -10  # illegal pickup

                            # Dropoff
                            elif action == 5:
                                if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
                                    new_pass_idx = dest_idx
                                    terminated = True
                                    reward = 20
                                elif (taxi_loc in self.locs) and pass_idx == 4:
                                    new_pass_idx = self.locs.index(taxi_loc)
                                else:
                                    reward = -10  # illegal dropoff

                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx
                            )
                            self.P[state][action].append(
                                (1.0, new_state, reward, terminated)
                            )

        # Normalize başlangıç dağılımı
        total = self.initial_state_distrib.sum()
        if total == 0:
            raise ValueError("initial_state_distrib boş, bir yerde hata var.")
        self.initial_state_distrib /= total

        self.s = 0
        self.lastaction = None

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        i = taxi_row
        i *= self.num_cols
        i += taxi_col
        i *= self.num_pass_locs
        i += pass_loc
        i *= self.num_dest_locs
        i += dest_idx
        return i

    def decode(self, i):
        dest_idx = i % self.num_dest_locs
        i //= self.num_dest_locs
        pass_loc = i % self.num_pass_locs
        i //= self.num_pass_locs
        taxi_col = i % self.num_cols
        taxi_row = i // self.num_cols
        return taxi_row, taxi_col, pass_loc, dest_idx

    def valid_state(self, state: int) -> bool:
        taxi_row, taxi_col, _, _ = self.decode(state)
        return (taxi_row, taxi_col) not in self.blocked_cells

    def action_mask(self, state: int):
        mask = np.zeros(self.num_actions, dtype=np.int8)
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)

        # Eğer zaten blocked bir hücredeyse (teorik olarak olmamalı) tüm mask 0 kalsın
        if (taxi_row, taxi_col) in self.blocked_cells:
            return mask

        # Olası hareketler (bloklu hücreye gitmeye izin yok)
        if taxi_row < self.num_rows - 1 and (taxi_row + 1, taxi_col) not in self.blocked_cells:
            mask[0] = 1  # south
        if taxi_row > 0 and (taxi_row - 1, taxi_col) not in self.blocked_cells:
            mask[1] = 1  # north
        if taxi_col < self.num_cols - 1 and (taxi_row, taxi_col + 1) not in self.blocked_cells:
            mask[2] = 1  # east
        if taxi_col > 0 and (taxi_row, taxi_col - 1) not in self.blocked_cells:
            mask[3] = 1  # west

        if pass_loc < 4 and (taxi_row, taxi_col) == self.locs[pass_loc]:
            mask[4] = 1  # pickup

        if pass_loc == 4 and (
            (taxi_row, taxi_col) == self.locs[dest_idx]
            or (taxi_row, taxi_col) in self.locs
        ):
            mask[5] = 1  # dropoff

        return mask

    def step(self, action):
        transitions = self.P[self.s][action]
        p, s, r, terminated = transitions[0]
        self.s = s
        self.lastaction = action
        return (
            int(s),
            r,
            terminated,
            False,
            {"prob": p, "action_mask": self.action_mask(s)},
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # blocked olmayan bir başlangıç state'i sample et
        while True:
            s = np.random.choice(self.num_states, p=self.initial_state_distrib)
            if self.valid_state(s):
                self.s = s
                break
        self.lastaction = None
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}

    def render(self, mode="ansi"):
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(self.s)
        txt = f"Taxi: ({taxi_row},{taxi_col}) | passenger_loc={pass_loc} | dest={dest_idx}"
        print(txt)
        return txt

    def close(self):
        pass
