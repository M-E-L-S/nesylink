/-!
  这是Task2的针对性lean建模和策略证明。
  对应task2 - 状态机/BFS搜索agent的实现。
  完整包含了Task2涉及到的环境建模，证明了安全性、可达性、BFS完备性。
-/

namespace Task2Formalization

/-!
  基本环境建模
-/
abbrev Position := Nat × Nat

inductive Action where
  | wait | up | down | left | right | interact | attack
  deriving DecidableEq, Repr

structure SymbolicState where
  player : Position
  chests : List Position
  keys : Nat
  monsterHp : Nat
  deriving DecidableEq, Repr

def manhattan (a b : Position) : Nat :=
  let dx := if a.1 ≤ b.1 then b.1 - a.1 else a.1 - b.1
  let dy := if a.2 ≤ b.2 then b.2 - a.2 else a.2 - b.2
  dx + dy

def adjacent (a b : Position) : Prop := manhattan a b = 1

instance (a b : Position) : Decidable (adjacent a b) :=
  inferInstanceAs (Decidable (manhattan a b = 1))

def applyMove (p : Position) (a : Action) : Position :=
  match a with
  | Action.up => (p.1, if p.2 == 0 then 0 else p.2 - 1)
  | Action.down => (p.1, if p.2 == 7 then 7 else p.2 + 1)
  | Action.left => (if p.1 == 0 then 0 else p.1 - 1, p.2)
  | Action.right => (if p.1 == 9 then 9 else p.1 + 1, p.2)
  | _ => p

def isTrapB (p : Position) : Bool :=
-- task2陷阱硬编码
  if p.2 == 0 then p.1 >= 1 && p.1 <= 8
  else if p.2 == 7 then p.1 >= 1 && p.1 <= 8
  else false

def isSafeTileB (p : Position) : Bool :=
  decide (p.1 < 10) && decide (p.2 < 8) && !(isTrapB p)

def isSafeTile (p : Position) : Prop := isSafeTileB p = true

def t2MonsterPos : Position := (2, 2)

def canOpenChest (s : SymbolicState) (c : Position) : Prop :=
  c ∈ s.chests ∧ adjacent s.player c

def canAttackMonster (s : SymbolicState) : Prop :=
  s.monsterHp > 0 ∧ adjacent s.player t2MonsterPos

inductive Step : SymbolicState → Action → SymbolicState → Prop where
  | move
      {s : SymbolicState} {a : Action} :
      a ∈ [Action.up, Action.down, Action.left, Action.right] →
      isSafeTile (applyMove s.player a) →
      Step s a { s with player := applyMove s.player a }
  | moveBlocked
      {s : SymbolicState} {a : Action} :
      a ∈ [Action.up, Action.down, Action.left, Action.right] →
      ¬ isSafeTile (applyMove s.player a) →
      Step s a s
  | openChest
      {s : SymbolicState} {c : Position} :
      canOpenChest s c →
      Step s Action.interact { s with chests := s.chests.erase c, keys := s.keys + 1 }
  | attack
      {s : SymbolicState} :
      canAttackMonster s →
      Step s Action.attack { s with monsterHp := s.monsterHp - 1 }
  | actionNoEffect
      {s : SymbolicState} {a : Action} :
      a ∈ [Action.interact, Action.attack] →
      (a = Action.interact → ∀ c, ¬ canOpenChest s c) →
      (a = Action.attack → ¬ canAttackMonster s) →
      Step s a s
  | wait {s : SymbolicState} : Step s Action.wait s

inductive Exec : SymbolicState → List Action → SymbolicState → Prop where
  | nil {s : SymbolicState} : Exec s [] s
  | cons {s t u : SymbolicState} {a : Action} {rest : List Action} :
      Step s a t → Exec t rest u → Exec s (a :: rest) u

-- Target from agent.py: West exit (x=0). Requirements: keys > 0 AND monsters defeated.
def GoalReached (s : SymbolicState) : Prop :=
  s.keys > 0 ∧ s.monsterHp == 0 ∧ s.player.1 == 0

def TaskCompletable (s : SymbolicState) : Prop :=
  ∃ plan final, Exec s plan final ∧ GoalReached final

def SafeState (s : SymbolicState) : Prop :=
  isSafeTile s.player

/-!
  Part 1: 安全性
-/

theorem safe_move_preserves_safe_state
    {s t : SymbolicState} {a : Action}
    (h : Step s a t)
    (ha : a ∈ [Action.up, Action.down, Action.left, Action.right])
    (hsafe : isSafeTile (applyMove s.player a)) :
    SafeState t := by
  cases h with
  | move _ hsafe' => exact hsafe'
  | moveBlocked _ hblocked => exact False.elim (hblocked hsafe)
  | openChest => cases ha <;> contradiction
  | attack => cases ha <;> contradiction
  | actionNoEffect => cases a <;> simp_all
  | wait => cases ha <;> contradiction


/-!
  Part 2: BFS完备性
-/

def BoundedReachable (init : SymbolicState) (n : Nat) (target : SymbolicState) : Prop :=
  ∃ plan, plan.length ≤ n ∧ Exec init plan target

def BoundedGoalReachable (init : SymbolicState) (n : Nat) : Prop :=
  ∃ final, BoundedReachable init n final ∧ GoalReached final

def BfsFrontierComplete (init : SymbolicState) (n : Nat) (frontier : List SymbolicState) : Prop :=
  ∀ final, BoundedReachable init n final → final ∈ frontier

def BfsFindsGoal (frontier : List SymbolicState) : Prop :=
  ∃ final, final ∈ frontier ∧ GoalReached final

theorem bfs_completeness_from_frontier_invariant
    {init : SymbolicState} {n : Nat} {frontier : List SymbolicState}
    (hcomplete : BfsFrontierComplete init n frontier)
    (hreachable : BoundedGoalReachable init n) :
    BfsFindsGoal frontier := by
  rcases hreachable with ⟨final, hbounded, hgoal⟩
  exact ⟨final, hcomplete final hbounded, hgoal⟩

def adjacentB (p1 p2 : Position) : Bool := manhattan p1 p2 == 1

def findTarget? (xs : List Position) (p : Position → Bool) : Option Position :=
  match xs with
  | [] => none
  | x :: rest => if p x then some x else findTarget? rest p

def canOpenChestB (s : SymbolicState) (c : Position) : Bool :=
  adjacentB s.player c

def symbolicStepFn (s : SymbolicState) : Action → SymbolicState
  | Action.up =>
      let rp := applyMove s.player Action.up
      if isSafeTileB rp then { s with player := rp } else s
  | Action.down =>
      let rp := applyMove s.player Action.down
      if isSafeTileB rp then { s with player := rp } else s
  | Action.left =>
      let rp := applyMove s.player Action.left
      if isSafeTileB rp then { s with player := rp } else s
  | Action.right =>
      let rp := applyMove s.player Action.right
      if isSafeTileB rp then { s with player := rp } else s
  | Action.interact =>
      match findTarget? s.chests (canOpenChestB s) with
      | some c => { s with chests := s.chests.erase c, keys := s.keys + 1 }
      | none => s
  | Action.attack =>
      if s.monsterHp > 0 && adjacentB s.player t2MonsterPos then
        { s with monsterHp := s.monsterHp - 1 }
      else s
  | _ => s

def runPlanFn : SymbolicState → List Action → SymbolicState
  | s, [] => s
  | s, a :: rest => runPlanFn (symbolicStepFn s a) rest

def allActions : List Action :=
  [Action.wait, Action.up, Action.down, Action.left, Action.right, Action.interact, Action.attack]

theorem action_mem_allActions (a : Action) : a ∈ allActions := by cases a <;> simp [allActions]

def expandFrontier : List SymbolicState → List SymbolicState
  | [] => []
  | s :: rest => allActions.map (symbolicStepFn s) ++ expandFrontier rest

def bfsVisited : Nat → List SymbolicState → List SymbolicState
  | 0, frontier => frontier
  | n + 1, frontier => frontier ++ bfsVisited n (expandFrontier frontier)

def bfsVisitedFrom (init : SymbolicState) (n : Nat) : List SymbolicState :=
  bfsVisited n [init]

theorem mem_expandFrontier_of_mem {s : SymbolicState} {frontier : List SymbolicState} (a : Action) (hs : s ∈ frontier) :
    symbolicStepFn s a ∈ expandFrontier frontier := by
  induction frontier with
  | nil => cases hs
  | cons head tail ih =>
      simp [expandFrontier] at hs ⊢
      rcases hs with hEq | hTail
      · subst hEq; exact Or.inl ⟨a, action_mem_allActions a, rfl⟩
      · exact Or.inr (ih hTail)

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
          have hStepMem : symbolicStepFn s head ∈ expandFrontier frontier := mem_expandFrontier_of_mem head hs
          have hRest := ih tail hStepMem hRestLen
          simp [runPlanFn, bfsVisited]
          exact Or.inr hRest

def FnBoundedReachable (init : SymbolicState) (n : Nat) (target : SymbolicState) : Prop :=
  ∃ plan, plan.length ≤ n ∧ runPlanFn init plan = target

def FnBfsFrontierComplete (init : SymbolicState) (n : Nat) (visited : List SymbolicState) : Prop :=
  ∀ target, FnBoundedReachable init n target → target ∈ visited

theorem bfsVisited_frontier_invariant (init : SymbolicState) (n : Nat) :
    FnBfsFrontierComplete init n (bfsVisitedFrom init n) := by
  intro target hreachable
  rcases hreachable with ⟨plan, hlen, hrun⟩
  rw [← hrun]
  exact runPlanFn_mem_bfsVisited_of_mem plan (by simp) hlen

/-!
  Part 3: 可达性
-/

theorem exec_append
    {s t u : SymbolicState} {p q : List Action}
    (hp : Exec s p t) (hq : Exec t q u) : Exec s (p ++ q) u := by
  induction hp with
  | nil => exact hq
  | cons hstep _ ih => exact Exec.cons hstep (ih hq)

theorem step_move_compute {s : SymbolicState} {a : Action} {pos' : Position}
    (ha : a ∈ [Action.up, Action.down, Action.left, Action.right])
    (hcomp : applyMove s.player a = pos')
    (hsafe : isSafeTileB pos' = true) :
    Step s a { s with player := pos' } := by
  have h_base : Step s a { s with player := applyMove s.player a } := by
    apply Step.move ha
    unfold isSafeTile
    rw [hcomp]
    exact hsafe
  rw [hcomp] at h_base
  exact h_base

theorem step_attack_compute {s : SymbolicState}
    (hhp : s.monsterHp > 0)
    (hadj : adjacent s.player t2MonsterPos) :
    Step s Action.attack { s with monsterHp := s.monsterHp - 1 } := by
  apply Step.attack
  exact ⟨hhp, hadj⟩

theorem step_interact_compute {s : SymbolicState} {c : Position}
    (hc : c ∈ s.chests)
    (hadj : adjacent s.player c) :
    Step s Action.interact { s with chests := s.chests.erase c, keys := s.keys + 1 } := by
  apply Step.openChest
  exact ⟨hc, hadj⟩

def t2Init : SymbolicState :=
  { player := (7, 3), chests := [(1, 3)], keys := 0, monsterHp := 2 }

-- 构造可行解
def t2ToMonster : List Action := [Action.left, Action.left, Action.left, Action.left, Action.up]
def t2NearMonster : SymbolicState := { t2Init with player := (3, 2) }

theorem t2_exec_to_monster : Exec t2Init t2ToMonster t2NearMonster := by
  unfold t2ToMonster
  iterate 5 (apply Exec.cons; exact step_move_compute (by decide) rfl (by decide))
  exact Exec.nil

def t2Attacks : List Action := [Action.attack, Action.attack]
def t2AfterMonster : SymbolicState := { t2NearMonster with monsterHp := 0 }

theorem t2_exec_attacks : Exec t2NearMonster t2Attacks t2AfterMonster := by
  unfold t2Attacks
  apply Exec.cons
  · apply step_attack_compute (by decide) (by decide)
  · apply Exec.cons
    · apply step_attack_compute (by decide) (by decide)
    · exact Exec.nil

def t2ToChest : List Action := [Action.down, Action.left]
def t2NearChest : SymbolicState := { t2AfterMonster with player := (2, 3) }

theorem t2_exec_to_chest : Exec t2AfterMonster t2ToChest t2NearChest := by
  unfold t2ToChest
  iterate 2 (apply Exec.cons; exact step_move_compute (by decide) rfl (by decide))
  exact Exec.nil

def t2Interact : List Action := [Action.interact]
def t2AfterChest : SymbolicState := { t2NearChest with chests := [], keys := 1 }

theorem t2_exec_interact : Exec t2NearChest t2Interact t2AfterChest := by
  unfold t2Interact
  apply Exec.cons
  · apply step_interact_compute (c := (1, 3)) (by decide) (by decide)
  · exact Exec.nil

def t2ToExit : List Action := [Action.left, Action.left]
def t2Final : SymbolicState := { t2AfterChest with player := (0, 3) }

theorem t2_exec_to_exit : Exec t2AfterChest t2ToExit t2Final := by
  unfold t2ToExit
  iterate 2 (apply Exec.cons; exact step_move_compute (by decide) rfl (by decide))
  exact Exec.nil

def t2FullPlan : List Action :=
  t2ToMonster ++ t2Attacks ++ t2ToChest ++ t2Interact ++ t2ToExit

theorem task2_concrete_completable : TaskCompletable t2Init := by
  have h1 := exec_append t2_exec_to_monster t2_exec_attacks
  have h2 := exec_append h1 t2_exec_to_chest
  have h3 := exec_append h2 t2_exec_interact
  have h4 := exec_append h3 t2_exec_to_exit
  exact ⟨t2FullPlan, t2Final, h4, by unfold GoalReached; decide⟩

end Task2Formalization