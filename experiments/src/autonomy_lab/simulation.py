from __future__ import annotations
import math, random
from collections import Counter, defaultdict, deque
import networkx as nx

L, M, R = 0, 1, 2
ACTIONS = [L, M, R]
ACTION_NAMES = {L: 'L', M: 'M', R: 'R'}
CONTENT = ['H','P','D']


def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
def clip(x, lo, hi): return max(lo, min(hi, x))
def state_value(a): return {L:0.0, M:1.0, R:2.0}[a]

def benefit(action, charisma, ctype='N'):
    base = {L:1.0, M:2.3, R:3.0}[action]
    if ctype == 'D' and action == R:
        base = 3.2
    if ctype == 'P' and action == M:
        base = 2.8
    return base * charisma

def punishment_cost(action, cost_scale):
    return {L:0.0, M:2.0, R:7.0}[action] * cost_scale

class QAgent:
    def __init__(self, alpha=0.10, gamma=0.95, epsilon=0.08):
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def act(self, obs):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        qs = [self.Q[(obs,a)] for a in ACTIONS]
        m = max(qs)
        best = [a for a,q in zip(ACTIONS,qs) if q == m]
        return random.choice(best)
    def update(self, s, a, r, s2):
        old = self.Q[(s,a)]
        best_next = max(self.Q[(s2,a2)] for a2 in ACTIONS)
        self.Q[(s,a)] = old + self.alpha * (r + self.gamma * best_next - old)


def reactive_policy(obs):
    """Myopic threshold heuristic: reacts to current observation without learning."""
    infl_bucket, alarm_bucket, punished, bridge = obs
    if punished:
        return L
    if alarm_bucket >= 2:
        return L
    if alarm_bucket == 1:
        return M
    if infl_bucket >= 2:
        return R
    if infl_bucket == 1:
        return M
    return random.choices(ACTIONS, weights=[0.30, 0.55, 0.15])[0]


FORCE_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

class RegulatorAgent:
    def __init__(self, alpha=0.05, gamma=0.95, epsilon=0.10):
        self.Q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def observe(self, alarm, radical_frac):
        alarm_b = 0 if alarm < 0.3 else (1 if alarm < 0.5 else (2 if alarm < 0.7 else 3))
        rad_b = 0 if radical_frac < 0.15 else (1 if radical_frac < 0.30 else 2)
        return (alarm_b, rad_b)

    def act(self, obs):
        if random.random() < self.epsilon:
            return random.choice(FORCE_LEVELS)
        qs = [self.Q[(obs, f)] for f in FORCE_LEVELS]
        m = max(qs)
        best = [f for f, q in zip(FORCE_LEVELS, qs) if q == m]
        return random.choice(best)

    def update(self, s, a, r, s2):
        old = self.Q[(s, a)]
        best_next = max(self.Q[(s2, f)] for f in FORCE_LEVELS)
        self.Q[(s, a)] = old + self.alpha * (r + self.gamma * best_next - old)


MULTIPLIER_LEVELS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

class MultiplierBandit:
    """EMA-based bandit that adapts bridge_multiplier from observed bridge error rates."""
    def __init__(self, levels=None, ema_alpha=0.05, epsilon=0.10):
        self.levels = levels or list(MULTIPLIER_LEVELS)
        self.epsilon = epsilon
        self.ema_alpha = ema_alpha
        self.reward_ema = {m: 0.0 for m in self.levels}
        self.counts = {m: 0 for m in self.levels}
        self.current = self.levels[0]

    def act(self):
        if random.random() < self.epsilon:
            self.current = random.choice(self.levels)
        else:
            self.current = max(self.levels, key=lambda m: self.reward_ema[m])
        return self.current

    def update(self, m, reward):
        self.counts[m] += 1
        self.reward_ema[m] += self.ema_alpha * (reward - self.reward_ema[m])


def build_graph(cfg, seed):
    random.seed(seed)
    n = int(cfg.get('n', 240))
    topology = cfg.get('topology','modular')
    G = nx.DiGraph()
    if topology == 'er':
        p = float(cfg.get('p_er', 0.02))
        UG = nx.erdos_renyi_graph(n, p, seed=seed, directed=False)
        for u,v in UG.edges():
            if random.random() < 0.75: G.add_edge(u,v)
            if random.random() < 0.75: G.add_edge(v,u)
        comm = {i: 0 for i in range(n)}
        bridges = set(sorted(range(n), key=lambda i: UG.degree(i), reverse=True)[:max(1,n//20)])
    elif topology == 'scale_free':
        G0 = nx.scale_free_graph(n, seed=seed)
        for u,v in G0.edges():
            if u != v: G.add_edge(int(u), int(v))
        for i in range(n):
            if i not in G: G.add_node(i)
        UG = G.to_undirected()
        comm = {i: 0 for i in range(n)}
        nodes_in_graph = list(UG.nodes())
        bridges = set(sorted(nodes_in_graph, key=lambda i: UG.degree(i), reverse=True)[:max(1,n//20)])
    else:
        c = int(cfg.get('num_communities',6))
        sizes = [n//c]*c
        for i in range(n % c): sizes[i] += 1
        p_in = float(cfg.get('p_in',0.08)); p_out = float(cfg.get('p_out',0.004))
        probs = [[p_in if i==j else p_out for j in range(c)] for i in range(c)]
        UG = nx.stochastic_block_model(sizes, probs, seed=seed, directed=False)
        start = 0; comm = {}
        for cc,sz in enumerate(sizes):
            for node in range(start,start+sz): comm[node] = cc
            start += sz
        for u,v in UG.edges():
            if random.random() < 0.75: G.add_edge(u,v)
            if random.random() < 0.75: G.add_edge(v,u)
        for i in range(n):
            if i not in G: G.add_node(i)
        bridges = set()
        frac = float(cfg.get('bridge_fraction',0.12))
        UG_undirected = G.to_undirected()
        bc = nx.betweenness_centrality(UG_undirected)
        n_bridges = max(1, int(n * frac))
        bridges = set(sorted(bc, key=lambda node: bc[node], reverse=True)[:n_bridges])
    for i in range(n):
        if i not in G: G.add_node(i)
    # Content placement: assign dangerous (D) content with optional bridge bias.
    # "uniform" (default) draws each node's type independently. "bridge_biased"
    # concentrates D on bridge nodes; "nonbridge_biased" concentrates D off bridges.
    # The overall D fraction (~0.20) is held fixed across placements.
    placement = cfg.get('content_placement', 'uniform')
    forced_ctype = {}
    if cfg.get('enable_content', False) and placement in ('bridge_biased', 'nonbridge_biased'):
        n_dangerous = int(round(0.20 * n))
        bridge_list = list(bridges)
        nonbridge_list = [i for i in range(n) if i not in bridges]
        random.shuffle(bridge_list); random.shuffle(nonbridge_list)
        primary = bridge_list + nonbridge_list if placement == 'bridge_biased' else nonbridge_list + bridge_list
        d_nodes = set(primary[:n_dangerous])
        remaining = [i for i in range(n) if i not in d_nodes]
        random.shuffle(remaining)
        # Split the rest between H (0.35/0.80) and P (0.45/0.80) proportionally
        n_h = int(round(0.35/0.80 * len(remaining)))
        for idx, node in enumerate(remaining):
            forced_ctype[node] = 'H' if idx < n_h else 'P'
        for node in d_nodes:
            forced_ctype[node] = 'D'
    for i in G.nodes():
        is_bridge = i in bridges
        if not cfg.get('enable_content', False):
            ctype = 'N'
        elif i in forced_ctype:
            ctype = forced_ctype[i]
        else:
            ctype = random.choices(CONTENT, weights=[0.35,0.45,0.20])[0]
        G.nodes[i].update({
            'action': random.choices(ACTIONS, weights=[0.80,0.17,0.03])[0],
            'charisma': random.uniform(0.85,1.25) + (0.2 if is_bridge else 0.0),
            'cost_scale': random.uniform(0.9,1.5),
            'detectability': clip(random.uniform(0.45,1.0) + (0.15 if is_bridge else 0.0),0,1.0),
            'community': comm.get(i,0),
            'is_bridge': is_bridge,
            'content_type': ctype,
            'predicted_type': ctype,
            'punished_last_step': False,
            'agent': QAgent(
                alpha=float(cfg.get('agent_alpha', 0.10)),
                gamma=float(cfg.get('agent_gamma', 0.95)),
                epsilon=float(cfg.get('agent_epsilon', 0.08)),
            ),
        })
    for u,v in G.edges():
        intra = G.nodes[u].get('community') == G.nodes[v].get('community')
        G[u][v]['weight'] = random.uniform(0.7,1.6) if intra else random.uniform(0.5,1.3)
    return G


def local_influence(G, i):
    return sum(G[j][i].get('weight',1.0) * state_value(G.nodes[j]['action']) for j in G.predecessors(i))

def global_alarm(G):
    return sum(G.nodes[i]['detectability']*state_value(G.nodes[i]['action']) for i in G.nodes()) / max(len(G),1)

def fractions(G):
    counts = Counter(G.nodes[i]['action'] for i in G.nodes())
    n = max(len(G),1)
    return {'L': counts[L]/n, 'M': counts[M]/n, 'R': counts[R]/n}

def usefulness(G):
    return sum(benefit(G.nodes[i]['action'], G.nodes[i]['charisma'], G.nodes[i].get('content_type','P')) for i in G.nodes()) / max(len(G),1)

def bridge_radical_fraction(G):
    b = [i for i in G.nodes() if G.nodes[i]['is_bridge']]
    if not b: return 0.0
    return sum(1 for i in b if G.nodes[i]['action'] == R) / len(b)

def dangerous_radical_bridge_fraction(G):
    b = [i for i in G.nodes() if G.nodes[i]['is_bridge']]
    if not b: return 0.0
    return sum(1 for i in b if G.nodes[i]['action'] == R and G.nodes[i]['content_type'] == 'D') / len(b)

CONFUSIONS = {
    'oracle': {'H':{'H':1,'P':0,'D':0}, 'P':{'H':0,'P':1,'D':0}, 'D':{'H':0,'P':0,'D':1}},
    'default': {'H':{'H':0.85,'P':0.13,'D':0.02}, 'P':{'H':0.20,'P':0.65,'D':0.15}, 'D':{'H':0.05,'P':0.20,'D':0.75}},
    'fp_heavy': {'H':{'H':0.70,'P':0.10,'D':0.20}, 'P':{'H':0.10,'P':0.55,'D':0.35}, 'D':{'H':0.05,'P':0.20,'D':0.75}},
    'fn_heavy': {'H':{'H':0.85,'P':0.13,'D':0.02}, 'P':{'H':0.20,'P':0.70,'D':0.10}, 'D':{'H':0.25,'P':0.50,'D':0.25}},
}

def _interpolate_confusion(alpha, base='fp_heavy'):
    """Interpolate between a base confusion matrix and oracle. alpha=1 → oracle, alpha=0 → base."""
    base_mat = CONFUSIONS[base]
    oracle = CONFUSIONS['oracle']
    result = {}
    for c in base_mat:
        result[c] = {}
        for label in base_mat[c]:
            result[c][label] = alpha * oracle[c][label] + (1 - alpha) * base_mat[c][label]
    return result

def sample_predicted_labels(G, noise_mode='oracle', classifier_accuracy=None):
    if classifier_accuracy is not None:
        mat = _interpolate_confusion(classifier_accuracy)
    else:
        mat = CONFUSIONS.get(noise_mode, CONFUSIONS['default'])
    for i in G.nodes():
        c = G.nodes[i]['content_type']
        probs = mat[c]
        labels = list(probs.keys()); weights = list(probs.values())
        G.nodes[i]['predicted_type'] = random.choices(labels, weights=weights)[0]

_CONTENT_TRANSITIONS = {
    R: {'H': {'H': 0.90, 'P': 0.07, 'D': 0.03},
        'P': {'H': 0.05, 'P': 0.80, 'D': 0.15},
        'D': {'H': 0.02, 'P': 0.08, 'D': 0.90}},
    M: {'H': {'H': 0.92, 'P': 0.06, 'D': 0.02},
        'P': {'H': 0.05, 'P': 0.88, 'D': 0.07},
        'D': {'H': 0.03, 'P': 0.12, 'D': 0.85}},
    L: {'H': {'H': 0.95, 'P': 0.04, 'D': 0.01},
        'P': {'H': 0.12, 'P': 0.83, 'D': 0.05},
        'D': {'H': 0.08, 'P': 0.15, 'D': 0.77}},
}

def build_content_transitions(cfg):
    strength = float(cfg.get('content_transition_strength', 1.0))
    if strength == 1.0:
        return _CONTENT_TRANSITIONS
    out = {}
    for action in [L, M, R]:
        out[action] = {}
        for src in CONTENT:
            row = dict(_CONTENT_TRANSITIONS[action][src])
            off = {k: v * strength for k, v in row.items() if k != src}
            off_sum = sum(off.values())
            if off_sum > 1.0:
                scale = 0.99 / off_sum
                off = {k: v * scale for k, v in off.items()}
                off_sum = sum(off.values())
            off[src] = 1.0 - off_sum
            out[action][src] = off
    return out

def apply_content_transitions(G, actions, transitions):
    for i in G.nodes():
        action = actions.get(i)
        if action is None:
            continue
        current = G.nodes[i]['content_type']
        if current == 'N':
            continue
        row = transitions[action][current]
        labels = list(row.keys())
        weights = list(row.values())
        G.nodes[i]['content_type'] = random.choices(labels, weights=weights)[0]


def fp_fn_metrics(G):
    nodes = list(G.nodes())
    bridges = [i for i in nodes if G.nodes[i]['is_bridge']]
    def rates(sub):
        n = max(len(sub),1)
        fp = sum(1 for i in sub if G.nodes[i]['predicted_type']=='D' and G.nodes[i]['content_type']!='D') / n
        fn = sum(1 for i in sub if G.nodes[i]['predicted_type']!='D' and G.nodes[i]['content_type']=='D') / n
        return fp, fn
    fp, fn = rates(nodes); bfp, bfn = rates(bridges)
    return fp, fn, bfp, bfn

def observe(G, i, delayed_alarm):
    infl = local_influence(G, i)
    alarm_bucket = 0 if delayed_alarm < 0.35 else (1 if delayed_alarm < 0.75 else 2)
    infl_bucket = 0 if infl < 2.0 else (1 if infl < 5.0 else 2)
    punished = int(G.nodes[i]['punished_last_step'])
    bridge = int(G.nodes[i]['is_bridge'])
    return (infl_bucket, alarm_bucket, punished, bridge)

def repression_prob(G, i, delayed_alarm, cfg, force, bridge_mult_override=None):
    base = sigmoid(float(cfg.get('k',10.0)) * (delayed_alarm - float(cfg.get('alarm_threshold',0.72))))
    mult = 1.0
    if G.nodes[i]['is_bridge'] and G.nodes[i].get('predicted_type') == 'D':
        mult = bridge_mult_override if bridge_mult_override is not None else float(cfg.get('bridge_multiplier',1.0))
    return clip(force * base * G.nodes[i]['detectability'] * mult, 0.0, 1.0)

def step(G, memory, cfg, rng, reg_agent=None, mult_bandit=None, content_transitions=None):
    delayed_alarm = memory[0]
    if cfg.get('enable_noise', False):
        sample_predicted_labels(G, cfg.get('noise_mode','default'), classifier_accuracy=cfg.get('classifier_accuracy'))
    else:
        for i in G.nodes(): G.nodes[i]['predicted_type'] = G.nodes[i]['content_type']
    reg_obs = None
    if reg_agent is not None:
        fr = fractions(G)
        reg_obs = reg_agent.observe(delayed_alarm, fr['R'])
        force = reg_agent.act(reg_obs)
    else:
        force = float(cfg.get('regulator_force', 1.0))
    old_obs, actions = {}, {}
    mode = cfg.get('learning_mode', 'q_learning' if cfg.get('learning', True) else 'fixed')
    for i in G.nodes():
        old_obs[i] = observe(G, i, delayed_alarm)
        if mode == 'q_learning':
            actions[i] = G.nodes[i]['agent'].act(old_obs[i])
        elif mode == 'reactive':
            actions[i] = reactive_policy(old_obs[i])
        else:
            actions[i] = random.choices(ACTIONS, weights=[0.65,0.27,0.08])[0]
    for i,a in actions.items(): G.nodes[i]['action'] = a
    if content_transitions is not None:
        apply_content_transitions(G, actions, content_transitions)
        if cfg.get('enable_noise', False):
            sample_predicted_labels(G, cfg.get('noise_mode','default'), classifier_accuracy=cfg.get('classifier_accuracy'))
        else:
            for i in G.nodes(): G.nodes[i]['predicted_type'] = G.nodes[i]['content_type']
    active_mult = None
    if mult_bandit is not None:
        active_mult = mult_bandit.act()
    rewards, punished_flags, rep_probs = {}, {}, {}
    for i in G.nodes():
        p = repression_prob(G, i, delayed_alarm, cfg, force, bridge_mult_override=active_mult)
        punished = random.random() < p
        effective_cost_scale = G.nodes[i]['cost_scale'] * float(cfg.get('cost_scale_override', 1.0))
        rewards[i] = benefit(G.nodes[i]['action'], G.nodes[i]['charisma'], G.nodes[i]['content_type']) + float(cfg.get('influence_weight',0.22))*local_influence(G,i)*state_value(G.nodes[i]['action']) - (punishment_cost(G.nodes[i]['action'], effective_cost_scale) if punished else 0.0)
        punished_flags[i] = punished; rep_probs[i] = p
    for i in G.nodes(): G.nodes[i]['punished_last_step'] = punished_flags[i]
    current_alarm = global_alarm(G)
    memory.append(current_alarm)
    if len(memory) > int(cfg.get('delay_steps',6)) + 1:
        memory.popleft()
    next_delayed = memory[0]
    if mode == 'q_learning':
        for i in G.nodes():
            G.nodes[i]['agent'].update(old_obs[i], actions[i], rewards[i], observe(G, i, next_delayed))
    if reg_agent is not None:
        fr_after = fractions(G)
        reg_reward = -fr_after['R'] - 0.1 * force
        reg_obs_next = reg_agent.observe(current_alarm, fr_after['R'])
        reg_agent.update(reg_obs, force, reg_reward, reg_obs_next)
    fp, fn, bfp, bfn = fp_fn_metrics(G)
    if mult_bandit is not None:
        drb = dangerous_radical_bridge_fraction(G)
        mult_bandit.update(active_mult, -bfp - 2.0 * drb)
    n_nodes = max(len(G), 1)
    content_counts = Counter(G.nodes[i]['content_type'] for i in G.nodes())
    return {
        'alarm_true': current_alarm, 'alarm_delayed': delayed_alarm,
        'avg_repression': sum(rep_probs.values())/max(len(rep_probs),1),
        'punished_fraction': sum(1 for x in punished_flags.values() if x)/max(len(punished_flags),1),
        'usefulness': usefulness(G),
        'regulator_force': force,
        'bridge_radical_fraction': bridge_radical_fraction(G),
        'dangerous_radical_bridge_fraction': dangerous_radical_bridge_fraction(G),
        'false_positive_rate': fp, 'false_negative_rate': fn,
        'false_positive_bridge_rate': bfp, 'false_negative_bridge_rate': bfn,
        'bridge_multiplier_used': active_mult if active_mult is not None else float(cfg.get('bridge_multiplier', 1.0)),
        'content_frac_H': content_counts.get('H', 0) / n_nodes,
        'content_frac_P': content_counts.get('P', 0) / n_nodes,
        'content_frac_D': content_counts.get('D', 0) / n_nodes,
    }

def run_simulation(cfg):
    seed = int(cfg.get('seed',1))
    random.seed(seed)
    G = build_graph(cfg, seed)
    reg_agent = RegulatorAgent() if cfg.get('regulator') == 'rl' else None
    mult_bandit = MultiplierBandit(
        ema_alpha=float(cfg.get('mult_bandit_ema_alpha', 0.05)),
        epsilon=float(cfg.get('mult_bandit_epsilon', 0.10)),
    ) if cfg.get('adaptive_multiplier', False) else None
    content_transitions = build_content_transitions(cfg) if cfg.get('enable_endogenous_content', False) else None
    init_alarm = global_alarm(G)
    memory = deque([init_alarm]*(int(cfg.get('delay_steps',6))+1), maxlen=int(cfg.get('delay_steps',6))+1)
    history_keys = ['L','M','R','alarm_true','alarm_delayed','avg_repression','punished_fraction',
                    'usefulness','regulator_force','bridge_radical_fraction','dangerous_radical_bridge_fraction',
                    'false_positive_rate','false_negative_rate','false_positive_bridge_rate','false_negative_bridge_rate',
                    'bridge_multiplier_used','content_frac_H','content_frac_P','content_frac_D']
    history = {k: [] for k in history_keys}
    for _ in range(int(cfg.get('steps',250))):
        fr = fractions(G)
        for key in ['L','M','R']: history[key].append(fr[key])
        out = step(G, memory, cfg, random, reg_agent=reg_agent, mult_bandit=mult_bandit, content_transitions=content_transitions)
        for key,val in out.items(): history[key].append(val)
    return history
