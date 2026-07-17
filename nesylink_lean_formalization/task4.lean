/-!
  这是Task4的针对性lean建模和策略证明。
  对应task4 - 状态机/BFS搜索agent的实现。
  完整包含了Task4涉及到的环境建模，证明了安全性、可达性、BFS完备性。
-/

namespace Task4Formalization

/-!
  基本环境建模
-/
abbrev Position := Nat × Nat

inductive Room where
  | west | center | north | east | south
  deriving DecidableEq, Repr

inductive Action where
  | wait | up | down | left | right | interact | attack
  deriving DecidableEq, Repr

structure SymbolicState where
  room : Room
  player : Position
  bridgeState : Nat
  hasKey : Bool
  hasSword : Bool
  monsterHp : Nat
  hasGold : Bool
  deriving DecidableEq, Repr

def manhattan (a b : Position) : Nat :=
  let dx := if a.1 ≤ b.1 then b.1 - a.1 else a.1 - b.1
  let dy := if a.2 ≤ b.2 then b.2 - a.2 else a.2 - b.2
  dx + dy

def applyMove (s : SymbolicState) (a : Action) : Room × Position :=
-- 硬编码了门房间传送机制
  match s.room, s.player, a with
  | Room.west, (9, 4), Action.right => (Room.center, (0, 4))
  | Room.center, (0, 4), Action.left => (Room.west, (9, 4))
  | Room.center, (4, 0), Action.up => (Room.north, (4, 7))
  | Room.north, (4, 7), Action.down => (Room.center, (4, 0))
  | Room.center, (9, 4), Action.right =>
      if s.hasKey then (Room.east, (0, 4)) else (s.room, s.player)
  | Room.east, (0, 4), Action.left => (Room.center, (9, 4))
  | Room.center, (4, 7), Action.down => (Room.south, (4, 0))
  | Room.south, (4, 0), Action.up => (Room.center, (4, 7))
  | r, (x, y), Action.up => (r, (x, if y == 0 then 0 else y - 1))
  | r, (x, y), Action.down => (r, (x, if y == 7 then 7 else y + 1))
  | r, (x, y), Action.left => (r, (if x == 0 then 0 else x - 1, y))
  | r, (x, y), Action.right => (r, (if x == 9 then 9 else x + 1, y))
  | r, p, _ => (r, p)

def isSafeTileB (r : Room) (p : Position) (bridge : Nat) : Bool :=
-- 这里硬编码了悬崖的地块
  if p.1 ≥ 10 ∨ p.2 ≥ 8 then false
  else if r == Room.center then
    if bridge == 0 then
      (p.1 ≤ 5 ∧ (p.2 == 3 ∨ p.2 == 4)) ∨ ((p.1 == 4 ∨ p.1 == 5) ∧ p.2 ≤ 4)
    else if bridge == 1 then
      p.2 == 3 ∨ p.2 == 4
    else
      (p.1 ≤ 5 ∧ (p.2 == 3 ∨ p.2 == 4)) ∨ ((p.1 == 4 ∨ p.1 == 5) ∧ p.2 ≥ 3)
  else true

def applyInteract (s : SymbolicState) : SymbolicState :=
-- 换桥机关
  match s.room, s.player with
  | Room.west, p =>
      if manhattan p (4, 4) ≤ 1 then { s with bridgeState := (s.bridgeState + 1) % 3 } else s
  | Room.north, p =>
      if manhattan p (4, 3) ≤ 1 then { s with hasKey := true } else s
  | Room.east, p =>
      if manhattan p (5, 4) ≤ 1 then { s with hasSword := true } else s
  | Room.center, p =>
      if s.monsterHp == 0 ∧ manhattan p (4, 4) ≤ 1 then { s with hasGold := true } else s
  | _, _ => s

def applyAttack (s : SymbolicState) : SymbolicState :=
  if s.room == Room.south ∧ s.monsterHp > 0 ∧ manhattan s.player (4, 4) ≤ 1 ∧ s.hasSword then
    { s with monsterHp := s.monsterHp - 1 }
  else s

def stepFn (s : SymbolicState) (a : Action) : SymbolicState :=
  match a with
  | Action.interact => applyInteract s
  | Action.attack => applyAttack s
  | Action.wait => s
  | _ =>
      if a ∈ [Action.up, Action.down, Action.left, Action.right] then
        let rp := applyMove s a
        if isSafeTileB rp.1 rp.2 s.bridgeState then
          { s with room := rp.1, player := rp.2 }
        else s
      else s

/-!
  Part 1: 安全性
-/

inductive Step : SymbolicState → Action → SymbolicState → Prop where
  | valid {s a} : Step s a (stepFn s a)

def SafeState (s : SymbolicState) : Prop :=
  isSafeTileB s.room s.player s.bridgeState = true

instance {s : SymbolicState} : Decidable (SafeState s) :=
  inferInstanceAs (Decidable (isSafeTileB s.room s.player s.bridgeState = true))

theorem safe_step_preserves_safe_state {s t : SymbolicState} {a : Action}
    (hsafe : SafeState s) (hstep : Step s a t) : SafeState t := by
  cases hstep
  unfold SafeState stepFn
  split
  · -- Action.interact
    unfold applyInteract
    split
    · rename_i p heq
      split
      · unfold SafeState at hsafe
        unfold isSafeTileB at hsafe ⊢
        simp_all
      · exact hsafe
    · rename_i p heq; split <;> exact hsafe
    · rename_i p heq; split <;> exact hsafe
    · rename_i p heq; split <;> exact hsafe
    · exact hsafe
  · -- Action.attack
    unfold applyAttack
    split <;> exact hsafe
  · -- Action.wait
    exact hsafe
  · -- 移动指令 (Action.up, down, left, right)
    dsimp only
    split
    · split
      · rename_i h_safe_tile
        exact h_safe_tile
      · exact hsafe
    · exact hsafe

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

def t4Init : SymbolicState :=
  { room := Room.west, player := (7, 4), bridgeState := 0,
    hasKey := false, hasSword := false, monsterHp := 1, hasGold := false }

def GoalReached (s : SymbolicState) : Prop :=
  s.hasGold = true ∧ s.room = Room.center ∧ s.player = (4, 4)

instance {s : SymbolicState} : Decidable (GoalReached s) :=
  inferInstanceAs (Decidable (s.hasGold = true ∧ s.room = Room.center ∧ s.player = (4, 4)))

def TaskCompletable (s : SymbolicState) : Prop :=
  ∃ plan final, Exec s plan final ∧ GoalReached final

-- 构造可行解
def t4Phase1_Key : List Action :=
  List.replicate 7 Action.right ++ List.replicate 9 Action.up ++ [Action.interact]

def t4Phase2_Bridge1 : List Action :=
  List.replicate 9 Action.down ++ List.replicate 10 Action.left ++ [Action.interact]

def t4Phase3_Sword : List Action :=
  List.replicate 21 Action.right ++ [Action.interact]

def t4Phase4_Bridge2 : List Action :=
  List.replicate 21 Action.left ++ [Action.interact]

def t4Phase5_Monster : List Action :=
  List.replicate 10 Action.right ++ List.replicate 7 Action.down ++ [Action.attack]

def t4Phase6_Victory : List Action :=
  List.replicate 7 Action.up ++ [Action.interact]

def t4FullPlan : List Action :=
  t4Phase1_Key ++ t4Phase2_Bridge1 ++ t4Phase3_Sword ++
  t4Phase4_Bridge2 ++ t4Phase5_Monster ++ t4Phase6_Victory

set_option maxRecDepth 4000

theorem task4_concrete_completable : TaskCompletable t4Init := by
  have hExec : Exec t4Init t4FullPlan (runPlanFn t4Init t4FullPlan) :=
    exec_of_runPlanFn rfl
  exact ⟨t4FullPlan, runPlanFn t4Init t4FullPlan, hExec, by decide⟩

end Task4Formalization