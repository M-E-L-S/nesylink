/-!
  这是Task5的针对性lean建模和策略证明。
  对应task5 - 状态机/BFS搜索agent的实现。
  完整包含了Task5涉及到的环境建模，证明了安全性、可达性、BFS完备性。
-/

namespace Task5Formalization

/-!
  基本环境建模
-/
abbrev Position := Nat × Nat

inductive Room where
  | center | south | east | west
  deriving DecidableEq, Repr

inductive Action where
  | wait | up | down | left | right | interact | attack
  deriving DecidableEq, Repr

structure SymbolicState where
  room : Room
  player : Position
  buttonPressed : Bool
  keys : Nat
  gold : Nat
  heals : Nat
  chestCenter : Bool
  chestSouth : Bool
  chestEast : Bool
  chestWest : Bool
  steps : Nat
  hp : Nat
  deriving DecidableEq, Repr

def manhattan (a b : Position) : Nat :=
  let dx := if a.1 ≤ b.1 then b.1 - a.1 else a.1 - b.1
  let dy := if a.2 ≤ b.2 then b.2 - a.2 else a.2 - b.2
  dx + dy

def isWall (r : Room) (p : Position) : Bool :=
-- 墙体硬编码
  match r, p with
  | Room.center, (5, 1) => true
  | Room.center, (5, 2) => true
  | Room.center, (3, 3) => true
  | Room.center, (4, 3) => true
  | Room.center, (6, 5) => true
  | Room.south, (2, 2) => true
  | Room.south, (3, 2) => true
  | Room.south, (4, 2) => true
  | Room.south, (5, 2) => true
  | Room.south, (6, 2) => true
  | Room.south, (7, 2) => true
  | Room.south, (4, 6) => true
  | Room.east, (2, 2) => true
  | Room.east, (2, 3) => true
  | Room.east, (2, 4) => true
  | Room.east, (5, 4) => true
  | Room.east, (6, 4) => true
  | Room.west, (1, 2) => true
  | Room.west, (2, 2) => true
  | Room.west, (5, 5) => true
  | Room.west, (4, 6) => true
  | Room.west, (5, 6) => true
  | _, _ => false

def isSafeTileB (r : Room) (p : Position) : Bool :=
  if p.1 ≥ 10 ∨ p.2 ≥ 8 then false
  else if r == Room.south ∧ p == (1, 5) then false -- 陷阱硬编码
  else if isWall r p then false
  else true

def applyInteract (s : SymbolicState) : SymbolicState :=
-- 翻译宝箱交互逻辑
  if s.room == Room.center then
    let (c1, g1) := if ¬s.chestCenter ∧ manhattan s.player (4, 2) ≤ 1 then (true, s.gold + 2) else (s.chestCenter, s.gold)
    { s with chestCenter := c1, gold := g1 }
  else if s.room == Room.south then
    let (c2, k1) := if ¬s.chestSouth ∧ manhattan s.player (8, 5) ≤ 1 then (true, s.keys + 1) else (s.chestSouth, s.keys)
    { s with chestSouth := c2, keys := k1 }
  else if s.room == Room.east then
    let (c3, h1) := if ¬s.chestEast ∧ manhattan s.player (7, 1) ≤ 1 then (true, s.heals + 1) else (s.chestEast, s.heals)
    { s with chestEast := c3, heals := h1 }
  else if s.room == Room.west then
    let (c4, g2) := if ¬s.chestWest ∧ manhattan s.player (2, 6) ≤ 1 then (true, s.gold + 5) else (s.chestWest, s.gold)
    { s with chestWest := c4, gold := g2 }
  else s

theorem applyInteract_safe (s : SymbolicState) (hs : isSafeTileB s.room s.player = true) :
    isSafeTileB (applyInteract s).room (applyInteract s).player = true := by
  unfold applyInteract
  split
  · exact hs
  · split
    · exact hs
    · split
      · exact hs
      · split
        · exact hs
        · exact hs

def applyTriggers (s : SymbolicState) : SymbolicState :=
-- 翻译了按钮的触发逻辑
  if s.room == Room.center ∧ s.player == (2, 6) then
    { s with buttonPressed := true }
  else s

theorem applyTriggers_safe (s : SymbolicState) :
    isSafeTileB (applyTriggers s).room (applyTriggers s).player = isSafeTileB s.room s.player := by
  unfold applyTriggers
  split
  · rfl
  · rfl

def getNextPosAndKeys (s : SymbolicState) (a : Action) : Room × Position × Nat :=
-- 翻译门的传送逻辑
  if a == Action.down ∧ s.room == Room.center ∧ s.player == (4, 7) ∧ s.buttonPressed then
    (Room.south, (4, 1), s.keys)
  else if a == Action.up ∧ s.room == Room.south ∧ s.player == (4, 0) then
    (Room.center, (4, 6), s.keys)
  else if a == Action.right ∧ s.room == Room.center ∧ s.player == (9, 4) ∧ s.keys > 0 then
    (Room.east, (1, 4), s.keys - 1)
  else if a == Action.left ∧ s.room == Room.east ∧ s.player == (0, 4) then
    (Room.center, (8, 4), s.keys)
  else if a == Action.left ∧ s.room == Room.center ∧ s.player == (0, 4) then
    (Room.west, (8, 4), s.keys)
  else if a == Action.right ∧ s.room == Room.west ∧ s.player == (9, 4) then
    (Room.center, (1, 4), s.keys)
  else
    let p := s.player
    let np := match a with
      | Action.up => (p.1, if p.2 == 0 then 0 else p.2 - 1)
      | Action.down => (p.1, if p.2 == 7 then 7 else p.2 + 1)
      | Action.left => (if p.1 == 0 then 0 else p.1 - 1, p.2)
      | Action.right => (if p.1 == 9 then 9 else p.1 + 1, p.2)
      | _ => p
    (s.room, np, s.keys)

def stepFn (s : SymbolicState) (a : Action) : SymbolicState :=
  let nxt_steps := s.steps + 1
  let nxt_hp := if nxt_steps % 33 == 0 then s.hp - 1 else s.hp
  -- 因为lean按格子简化，而实际游戏每6步移动一格，所以这里的200步扣血折算为约33步扣血。
  let s_time := { s with steps := nxt_steps, hp := nxt_hp }
  let s_act := match a with
  | Action.interact => applyInteract s_time
  | Action.attack => s_time
  | Action.wait => s_time
  | _ =>
      let (nr, np, nk) := getNextPosAndKeys s_time a
      if h : isSafeTileB nr np then
        { s_time with room := nr, player := np, keys := nk }
      else
        s_time
  applyTriggers s_act

/-!
  Part 1: 安全性
-/

inductive Step : SymbolicState → Action → SymbolicState → Prop where
  | valid {s a} : Step s a (stepFn s a)

def SafeState (s : SymbolicState) : Prop :=
  isSafeTileB s.room s.player = true

instance {s : SymbolicState} : Decidable (SafeState s) :=
  inferInstanceAs (Decidable (isSafeTileB s.room s.player = true))

theorem safe_step_preserves_safe_state {s t : SymbolicState} {a : Action}
    (hs : SafeState s) (hstep : Step s a t) : SafeState t := by
  cases hstep
  unfold SafeState at hs ⊢
  unfold stepFn
  rw [applyTriggers_safe]
  split
  · exact applyInteract_safe _ hs  -- 1. 交互动作分支
  · exact hs                       -- 2. 攻击动作分支
  · exact hs                       -- 3. 等待动作分支
  · -- 4. 移动动作分支 (_)
    split
    split
    · assumption
    · exact hs

/-!
  Part 2: BFS完备性
-/

inductive Exec : SymbolicState → List Action → SymbolicState → Prop where
  | nil {s : SymbolicState} : Exec s [] s
  | cons {s t u : SymbolicState} {a : Action} {rest : List Action} :
      Step s a t → Exec t rest u → Exec s (a :: rest) u

def BoundedReachable (init : SymbolicState) (n : Nat) (target : SymbolicState) : Prop :=
  ∃ plan, plan.length ≤ n ∧ Exec init plan target

def BfsFrontierComplete (init : SymbolicState) (n : Nat) (frontier : List SymbolicState) : Prop :=
  ∀ final, BoundedReachable init n final → final ∈ frontier

def allActions : List Action :=
  [Action.wait, Action.up, Action.down, Action.left, Action.right, Action.interact, Action.attack]

theorem action_mem_allActions (a : Action) : a ∈ allActions := by cases a <;> simp [allActions]

def expandFrontier : List SymbolicState → List SymbolicState
  | [] => []
  | s :: rest => allActions.map (stepFn s) ++ expandFrontier rest

def bfsVisited : Nat → List SymbolicState → List SymbolicState
  | 0, frontier => frontier
  | n + 1, frontier => frontier ++ bfsVisited n (expandFrontier frontier)

def bfsVisitedFrom (init : SymbolicState) (n : Nat) : List SymbolicState :=
  bfsVisited n [init]

theorem mem_expandFrontier_of_mem {s : SymbolicState} {frontier : List SymbolicState} (a : Action) (hs : s ∈ frontier) :
    stepFn s a ∈ expandFrontier frontier := by
  induction frontier with
  | nil => cases hs
  | cons head tail ih =>
      simp [expandFrontier] at hs ⊢
      rcases hs with hEq | hTail
      · subst hEq; exact Or.inl ⟨a, action_mem_allActions a, rfl⟩
      · exact Or.inr (ih hTail)

def runPlanFn : SymbolicState → List Action → SymbolicState
  | s, [] => s
  | s, a :: rest => runPlanFn (stepFn s a) rest

theorem runPlanFn_mem_bfsVisited_of_mem {s : SymbolicState} {frontier : List SymbolicState} (plan : List Action) {n : Nat}
    (hs : s ∈ frontier) (hlen : plan.length ≤ n) : runPlanFn s plan ∈ bfsVisited n frontier := by
  induction n generalizing s frontier plan with
  | zero =>
      cases plan with
      | nil => simpa [runPlanFn, bfsVisited] using hs
      | cons head tail => cases hlen
  | succ n ih =>
      cases plan with
      | nil => simp [runPlanFn, bfsVisited, hs]
      | cons head tail =>
          have hRestLen : tail.length ≤ n := Nat.le_of_succ_le_succ hlen
          have hStepMem : stepFn s head ∈ expandFrontier frontier := mem_expandFrontier_of_mem head hs
          have hRest := ih tail hStepMem hRestLen
          simp [runPlanFn, bfsVisited]
          exact Or.inr hRest

theorem exec_to_runPlanFn {s t : SymbolicState} {plan : List Action} (h : Exec s plan t) : runPlanFn s plan = t := by
  induction h with
  | nil => rfl
  | cons hstep _ ih =>
      cases hstep
      exact ih

theorem exec_of_runPlanFn {s : SymbolicState} {plan : List Action} {target : SymbolicState}
    (h : runPlanFn s plan = target) : Exec s plan target := by
  rw [← h]
  induction plan generalizing s with
  | nil => exact Exec.nil
  | cons a rest ih => exact Exec.cons Step.valid (ih h)

theorem bfs_frontier_complete (init : SymbolicState) (n : Nat) :
    BfsFrontierComplete init n (bfsVisitedFrom init n) := by
  intro final hreachable
  rcases hreachable with ⟨plan, hlen, hexec⟩
  have hrun := exec_to_runPlanFn hexec
  rw [← hrun]
  have hs : init ∈ [init] := by simp
  exact runPlanFn_mem_bfsVisited_of_mem plan hs hlen

/-!
  Part 3: 可达性
-/

def t5Init : SymbolicState :=
  { room := Room.center, player := (1, 1), buttonPressed := false,
    keys := 0, gold := 0, heals := 0,
    chestCenter := false, chestSouth := false, chestEast := false, chestWest := false,
    steps := 0, hp := 5 }

def GoalReached (s : SymbolicState) : Prop :=
  s.chestWest = true ∧ s.gold = 7 ∧ s.room = Room.west ∧ s.hp > 0

instance {s : SymbolicState} : Decidable (GoalReached s) :=
  inferInstanceAs (Decidable (s.chestWest = true ∧ s.gold = 7 ∧ s.room = Room.west ∧ s.hp > 0))

def TaskCompletable (s : SymbolicState) : Prop :=
  ∃ plan final, Exec s plan final ∧ GoalReached final

-- 构造可行解
def t5Phase1_CenterChest : List Action :=
  List.replicate 3 Action.right ++ [Action.down, Action.interact]

def t5Phase2_PressButton : List Action :=
  List.replicate 2 Action.left ++ List.replicate 4 Action.down

def t5Phase3_GoSouth : List Action :=
  List.replicate 2 Action.right ++ [Action.down, Action.down]

def t5Phase4_SouthChest : List Action :=
  List.replicate 4 Action.right ++ List.replicate 4 Action.down ++ [Action.interact]

def t5Phase5_GoNorth : List Action :=
  List.replicate 4 Action.up ++ List.replicate 4 Action.left ++ [Action.up, Action.up]

def t5Phase6_GoEast : List Action :=
  List.replicate 2 Action.up ++ List.replicate 5 Action.right ++ [Action.right]

def t5Phase7_EastChest : List Action :=
  List.replicate 3 Action.up ++ List.replicate 6 Action.right ++ [Action.interact]

def t5Phase8_ReturnCenter : List Action :=
  List.replicate 7 Action.left ++ List.replicate 3 Action.down ++ [Action.left]

def t5Phase9_GoWest : List Action :=
  List.replicate 8 Action.left ++ [Action.left]

def t5Phase10_WestChest : List Action :=
  List.replicate 6 Action.left ++ List.replicate 2 Action.down ++ [Action.interact]

def t5FullPlan : List Action :=
  t5Phase1_CenterChest ++ t5Phase2_PressButton ++ t5Phase3_GoSouth ++ t5Phase4_SouthChest ++
  t5Phase5_GoNorth ++ t5Phase6_GoEast ++ t5Phase7_EastChest ++ t5Phase8_ReturnCenter ++
  t5Phase9_GoWest ++ t5Phase10_WestChest

set_option maxRecDepth 6000
theorem task5_concrete_completable : TaskCompletable t5Init := by
  have hExec : Exec t5Init t5FullPlan (runPlanFn t5Init t5FullPlan) :=
    exec_of_runPlanFn rfl
  exact ⟨t5FullPlan, runPlanFn t5Init t5FullPlan, hExec, by decide⟩

end Task5Formalization