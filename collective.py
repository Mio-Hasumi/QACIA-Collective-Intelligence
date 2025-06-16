import pygame
import numpy as np
import random
import math
import imageio
from collections import deque
from PIL import Image

# === CONFIGURATION ===
SIMULATION_STEPS    = 1000
NUM_AGENTS          = 30
NUM_FOOD            = 10
AGENT_SIZE          = 5
SIM_WIDTH, SIM_HEIGHT = 600, 400

# Flocking weights
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT  = 1.0
COHESION_WEIGHT   = 1.0

# Predator parameters
PREDATOR_SPEED      = 3.0
FEAR_RADIUS         = 100
FLEE_WEIGHT         = 2.0
PREDATOR_START_STEP = 200

# Food/foraging parameters
FOOD_SENSE_RADIUS = 100
EAT_RADIUS        = 8
FOOD_ENERGY       = 1.0  # per food
PRUNE_AGE         = 100  # forget old food tips

# Memory & communication
MEMORY_SIZE       = 20        # predator memory
COMM_RADIUS       = 80        # for culture & food sharing
CULTURE_DIM       = 5
TRUST_INIT        = 0.5
TRUST_LEARN_RATE  = 0.05
TRUST_DECAY       = 0.995

# === HELPERS ===
def limit_vector(vec, max_magnitude):
    mag = np.linalg.norm(vec)
    return vec / mag * max_magnitude if mag > max_magnitude else vec

def dist(a, b):
    return np.linalg.norm(a - b)

# === FOOD CLASS ===
class Food:
    def __init__(self, x, y):
        self.pos    = np.array([x, y], dtype=float)
        self.energy = FOOD_ENERGY
    def draw(self, surf):
        if self.energy > 0:
            pygame.draw.circle(surf, (0,200,0), self.pos.astype(int), 4)

# === AGENT ===
class Agent:
    def __init__(self, x, y, idx, family_id):
        self.idx       = idx
        self.pos       = np.array([x, y], dtype=float)
        self.vel       = np.random.randn(2) * 0.5
        self.acc       = np.zeros(2)
        self.family_id = family_id

        # predator memory
        self.pred_mem  = deque(maxlen=MEMORY_SIZE)

        # food memory: fid -> (pos, last_seen_step)
        self.food_mem  = {}

        # trust and culture
        self.trust     = {j: TRUST_INIT for j in range(NUM_AGENTS) if j!=idx}
        self.culture   = np.random.rand(CULTURE_DIM)

        # foraging strategies
        self.strategy  = {
            'boldness': random.random(),      # 0=shy, 1=bold
            'explore_rate': random.random()   # prob of random roam
        }
        self.food_collected = 0

        self.color     = self.get_family_color()

    def get_family_color(self):
        cols = [(255,100,100),(100,255,100),(100,100,255),
                (255,255,100),(255,100,255)]
        return cols[self.family_id % len(cols)]

    def apply_force(self, f):
        self.acc += f

    # ——— SENSING & MEMORIES ———
    def sense_predator(self, predator):
        if predator:
            self.pred_mem.append(predator.pos.copy())

    def sense_food(self, foods, step):
        for fid, food in enumerate(foods):
            if food.energy<=0: continue
            d = dist(self.pos, food.pos)
            if d < FOOD_SENSE_RADIUS:
                self.food_mem[fid] = (food.pos.copy(), step)

    def prune_food_mem(self, step):
        self.food_mem = {
           fid:(pos,seen) 
           for fid,(pos,seen) in self.food_mem.items()
           if step - seen <= PRUNE_AGE
        }

    # ——— COMMUNICATION ———
    def communicate(self, others):
        for other in others:
            if other is self: continue
            if dist(self.pos, other.pos) < COMM_RADIUS:
                # foodexchange
                for fid,(pos,seen) in other.food_mem.items():
                    mine = self.food_mem.get(fid)
                    if mine is None or seen>mine[1]:
                        self.food_mem[fid] = (pos.copy(), seen)
                # culture blending
                delta = other.culture - self.culture
                w = self.trust[other.idx]
                self.culture += TRUST_LEARN_RATE * w * delta
                # strategy blending + mutation
                for k in self.strategy:
                    blend = (1-w)*self.strategy[k] + w*other.strategy[k]
                    self.strategy[k] = np.clip(blend + np.random.randn()*0.01, 0,1)
                # trust update by culture similarity
                sim = np.exp(-np.linalg.norm(delta))
                self.trust[other.idx] = TRUST_DECAY*self.trust[other.idx] + TRUST_LEARN_RATE*sim
                other.trust[self.idx] = TRUST_DECAY*other.trust[self.idx] + TRUST_LEARN_RATE*sim

    # ——— FLOCKING ———
    def separate(self, agents):
        steer, cnt = np.zeros(2),0
        for o in agents:
            if o is self: continue
            d = dist(self.pos,o.pos)
            if d<25 and d>0:
                steer += (self.pos-o.pos)/d
                cnt+=1
        return limit_vector(steer/cnt,0.1) if cnt else np.zeros(2)

    def align(self, agents):
        sumv,cnt = np.zeros(2),0
        for o in agents:
            if o is self: continue
            d = dist(self.pos,o.pos)
            neigh = 50 if o.family_id==self.family_id else 40
            if d<neigh:
                w = self.trust[o.idx]
                sumv += o.vel*w
                cnt += w
        if cnt:
            steer = (sumv/cnt)-self.vel
            return limit_vector(steer,0.1)
        return np.zeros(2)

    def cohere(self, agents):
        sump,tot= np.zeros(2),0
        for o in agents:
            if o is self: continue
            d=dist(self.pos,o.pos)
            neigh=50 if o.family_id==self.family_id else 40
            if d<neigh:
                fam = 1.5 if o.family_id==self.family_id else 1
                w = fam*self.trust[o.idx]
                sump += o.pos*w
                tot  += w
        if tot:
            steer = (sump/tot)-self.pos
            return limit_vector(steer,0.1)
        return np.zeros(2)

    def flock(self, agents):
        f = ( self.separate(agents)*SEPARATION_WEIGHT
            + self.align(agents)*ALIGNMENT_WEIGHT
            + self.cohere(agents)*COHESION_WEIGHT )
        self.apply_force(f)

    # ——— FORAGING ———
    def forage(self, foods):
        # decide: explore vs follow tip
        if self.food_mem and random.random() > self.strategy['explore_rate']:
            # pick highest-trusted tip
            best = max(self.food_mem.items(),
                       key=lambda x: self.trust.get(
                         # trust in the tipper is mixed in food_mem, assume uniform
                         # for simplicity
                         0, TRUST_INIT))
            target_pos,_ = best[1]
        else:
            # random wander scaled by boldness
            angle = random.random()*2*math.pi
            target_pos = self.pos + self.strategy['boldness']*np.array([math.cos(angle),math.sin(angle)])*50

        # steer toward target_pos
        desire = target_pos - self.pos
        if np.linalg.norm(desire)>0:
            steer = desire/np.linalg.norm(desire)*0.2
            self.apply_force(limit_vector(steer,0.1))

        # check eating
        for fid,(pos,_) in list(self.food_mem.items()):
            food = foods[fid]
            if food.energy>0 and dist(self.pos, food.pos)<EAT_RADIUS:
                # “eat”
                food.energy -= FOOD_ENERGY
                self.food_collected += FOOD_ENERGY
                # reward trust to all sources that mentioned this fid
                # here we give a flat bump to everyone
                for j in self.trust:
                    self.trust[j] = min(1.0, self.trust[j] + TRUST_LEARN_RATE)

    # ——— PREDATOR ———
    def flee_predator(self, predator):
        if not predator: return
        self.pred_mem.append(predator.pos.copy())
        d = dist(self.pos, predator.pos)
        if d<FEAR_RADIUS and d>0:
            steer = (self.pos - predator.pos)/d * (FEAR_RADIUS/d)
            self.apply_force(limit_vector(steer,0.5)*FLEE_WEIGHT)

    # ——— UPDATE & DRAW ———
    def update(self):
        self.vel += self.acc
        self.vel = limit_vector(self.vel, 2.0)
        self.pos += self.vel
        self.acc *= 0
        self.pos[0] %= SIM_WIDTH
        self.pos[1] %= SIM_HEIGHT

    def draw(self, surf):
        angle = math.atan2(self.vel[1], self.vel[0]) if np.linalg.norm(self.vel)>0 else 0
        pts = [(AGENT_SIZE*2,0),(-AGENT_SIZE,AGENT_SIZE),(-AGENT_SIZE,-AGENT_SIZE)]
        poly = [(
            self.pos[0]+px*math.cos(angle)-py*math.sin(angle),
            self.pos[1]+px*math.sin(angle)+py*math.cos(angle)
        ) for px,py in pts]
        pygame.draw.polygon(surf, self.color, poly)
        pygame.draw.polygon(surf, (0,0,0), poly,1)

# === PREDATOR ===
class Predator:
    def __init__(self, x,y):
        self.pos = np.array([x,y],dtype=float)
        self.vel = np.zeros(2)
    def update(self, agents):
        center = sum(a.pos for a in agents)/len(agents)
        dvec = center - self.pos
        if np.linalg.norm(dvec)>0:
            self.vel = dvec/np.linalg.norm(dvec)*PREDATOR_SPEED
        self.pos+=self.vel
        self.pos[0]%=SIM_WIDTH; self.pos[1]%=SIM_HEIGHT
    def draw(self,surf):
        pygame.draw.circle(surf,(200,0,0),self.pos.astype(int),8)

# === COLLECTIVE INTELLIGENCE METRIC ===
def measure_collective_intelligence(agents):
    # cohesion
    pos = np.array([a.pos for a in agents])
    center = pos.mean(axis=0)
    coh = 1.0/(1.0 + np.linalg.norm(pos-center,axis=1).mean())
    # trust
    trust_vals = [t for a in agents for t in a.trust.values()]
    tr  = np.mean(trust_vals)
    # memory consensus (predator)
    latest = np.array([a.pred_mem[-1] if a.pred_mem else np.zeros(2) for a in agents])
    mc  = np.exp(-np.linalg.norm(latest - latest.mean(axis=0),axis=1).mean()/FEAR_RADIUS)
    # foraging success
    food_tot = sum(a.food_collected for a in agents)/(NUM_AGENTS*FOOD_ENERGY)
    # combine
    return 0.3*coh + 0.3*tr + 0.2*mc + 0.2*food_tot

# === MAIN SIMULATION ===
def run_simulation():
    pygame.init()
    # spawn agents
    agents = [Agent(random.randint(0,SIM_WIDTH),
                    random.randint(0,SIM_HEIGHT),
                    i, i%5)
              for i in range(NUM_AGENTS)]
    # spawn food
    foods = [Food(random.randint(20,SIM_WIDTH-20),
                  random.randint(20,SIM_HEIGHT-20))
             for _ in range(NUM_FOOD)]
    predator = None
    frames = []

    for step in range(SIMULATION_STEPS):
        surf = pygame.Surface((SIM_WIDTH,SIM_HEIGHT))
        surf.fill((220,220,255))

        # draw & sense food
        for f in foods:
            f.draw(surf)
        for a in agents:
            a.sense_food(foods, step)
            a.prune_food_mem(step)

        # communication
        for a in agents:
            a.communicate(agents)

        # flocking + predator
        if step==PREDATOR_START_STEP:
            predator = Predator(0,0)
        if predator:
            predator.update(agents)
            for a in agents:
                a.flee_predator(predator)
            predator.draw(surf)
        for a in agents:
            a.flock(agents)

        # foraging
        for a in agents:
            a.forage(foods)

        # update & draw agents
        for a in agents:
            a.update()
            a.draw(surf)

        # capture
        view = pygame.surfarray.array3d(surf).transpose([1,0,2])
        frames.append(Image.fromarray(view))

        # print metrics
        if step%100==0:
            score = measure_collective_intelligence(agents)
            print(f"Step {step}: Intelligence = {score:.3f}")

    # save GIF
    imageio.mimsave('qacia_fullextended.gif', frames, fps=30)
    pygame.quit()

if __name__=='__main__':
    run_simulation()
