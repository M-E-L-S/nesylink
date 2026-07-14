/-!
  这是Task1的针对性lean建模和策略证明。
  对应task1 - 状态机/BFS搜索agent的实现。
  完整包含了Task1涉及到的环境建模，证明了安全性、可达性、BFS完备性。
-/

namespace Task1Formalization

/-!
  基本环境建模
-/
abbrev Position := Nat × Nat

inductive Action where
  | wait | up | down | left | right | interact
  deriving DecidableEq, Repr

structure SymbolicState where
  player : Position
  chests : List Position
  keys : Nat
  deriving DecidableEq, Repr

def manhattan (a b : Position) : Nat :=
  let dx := if a.1 ≤ b.1 then b.1 - a.1 else a.1 - b.1
  let dy := if a.2 ≤ b.2 then b.2 - a.2 else a.2 - b.2
  dx + dy

def adjacent (a b : Position) : Prop := manhattan a b = 1

def applyMove (p : Position) (a : Action) : Position :=
  match a with
  | Action.up => (p.1, if p.2 == 0 then 0 else p.2 - 1)
  | Action.down => (p.1, if p.2 == 7 then 7 else p.2 + 1)
  | Action.left => (if p.1 == 0 then 0 else p.1 - 1, p.2)
  | Action.right => (if p.1 == 9 then 9 else p.1 + 1, p.2)
  | _ => p

def isWallB (p : Position) : Bool :=
-- task1的墙面硬编码
  if p.2 == 2 then p.1 == 0 || p.1 == 1 || p.1 >= 4
  else if p.2 == 5 then p.1 <= 6
  else false

def isSafeTileB (p : Position) : Bool :=
  decide (p.1 < 10) && decide (p.2 < 8) && !(isWallB p)

def isSafeTile (p : Position) : Prop := isSafeTileB p = true

def adjacentChest (s : SymbolicState) : Prop :=
  ∃ cPos, cPos ∈ s.chests ∧ adjacent s.player cPos

def canOpenChest (s : SymbolicState) (c : Position) : Prop :=
  c ∈ s.chests ∧ adjacent s.player c

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
  | actionNoEffect
      {s : SymbolicState} {a : Action} :
      a = Action.interact →
      ¬ adjacentChest s →
      Step s a s
  | wait {s : SymbolicState} : Step s Action.wait s

inductive Exec : SymbolicState → List Action → SymbolicState → Prop where
  | nil {s : SymbolicState} : Exec s [] s
  | cons {s t u : SymbolicState} {a : Action} {rest : List Action} :
      Step s a t → Exec t rest u → Exec s (a :: rest) u

def GoalReached (s : SymbolicState) : Prop :=
  s.keys > 0 ∧ (s.player = (4, 0) ∨ s.player = (5, 0))

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

def posEqB (p1 p2 : Position) : Bool := p1.1 == p2.1 && p1.2 == p2.2
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
  | _ => s

def runPlanFn : SymbolicState → List Action → SymbolicState
  | s, [] => s
  | s, a :: rest => runPlanFn (symbolicStepFn s a) rest

def allActions : List Action :=
  [Action.wait, Action.up, Action.down, Action.left, Action.right, Action.interact]

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

def FnBoundedGoalReachable (init : SymbolicState) (n : Nat) : Prop :=
  ∃ final, FnBoundedReachable init n final ∧ GoalReached final

def FnBfsFindsGoal (visited : List SymbolicState) : Prop :=
  ∃ final, final ∈ visited ∧ GoalReached final

theorem bfsVisited_complete_for_bounded_goal {init : SymbolicState} {n : Nat}
    (hreachable : FnBoundedGoalReachable init n) : FnBfsFindsGoal (bfsVisitedFrom init n) := by
  rcases hreachable with ⟨final, hbounded, hgoal⟩
  exact ⟨final, bfsVisited_frontier_invariant init n final hbounded, hgoal⟩

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

theorem task1_completable_if_subplans_exist
    {init nearChest afterChest final : SymbolicState}
    {toChest toExit : List Action}
    {chest : Position}
    (hToChest : Exec init toChest nearChest)
    (hOpenable : canOpenChest nearChest chest)
    (hAfterChest : afterChest = { nearChest with chests := nearChest.chests.erase chest, keys := nearChest.keys + 1 })
    (hToExit : Exec afterChest toExit final)
    (hFinalKeys : final.keys > 0)
    (hFinalAtExit : final.player = (4, 0) ∨ final.player = (5, 0)) :
    TaskCompletable init := by
  have hOpenStep : Step nearChest Action.interact afterChest := by
    rw [hAfterChest]; exact Step.openChest hOpenable
  have hPhase1 : Exec init (toChest ++ [Action.interact]) afterChest :=
    exec_append hToChest (Exec.cons hOpenStep Exec.nil)
  have hAll : Exec init ((toChest ++ [Action.interact]) ++ toExit) final :=
    exec_append hPhase1 hToExit
  exact ⟨_, final, hAll, ⟨hFinalKeys, hFinalAtExit⟩⟩

def t1Init : SymbolicState :=
  { player := (4, 6), chests := [(0, 3)], keys := 0 }

def t1Chest : Position := (0, 3)

-- 构造可行解
def t1ToChest : List Action :=
  [Action.right, Action.right, Action.right, Action.up, Action.up, Action.up,
   Action.left, Action.left, Action.left, Action.left, Action.left, Action.left]

def t1ToExit : List Action :=
  [Action.right, Action.up, Action.up, Action.right, Action.right, Action.up]

def t1NearChest : SymbolicState := { t1Init with player := (1, 3) }
def t1AfterChest : SymbolicState := { t1NearChest with chests := [], keys := 1 }
def t1Final : SymbolicState := { t1AfterChest with player := (4, 0) }

theorem t1_exec_to_chest : Exec t1Init t1ToChest t1NearChest := by
  unfold t1ToChest
  iterate 12 (apply Exec.cons; exact step_move_compute (by decide) rfl (by decide))
  exact Exec.nil

theorem t1_exec_to_exit : Exec t1AfterChest t1ToExit t1Final := by
  unfold t1ToExit
  iterate 6 (apply Exec.cons; exact step_move_compute (by decide) rfl (by decide))
  exact Exec.nil

theorem task1_concrete_completable : TaskCompletable t1Init := by
  exact task1_completable_if_subplans_exist
    (init := t1Init)
    (nearChest := t1NearChest)
    (afterChest := t1AfterChest)
    (final := t1Final)
    (toChest := t1ToChest)
    (toExit := t1ToExit)
    (chest := t1Chest)
    t1_exec_to_chest
    (by unfold canOpenChest t1NearChest t1Init t1Chest adjacent manhattan; decide)
    rfl
    t1_exec_to_exit
    (by decide)
    (by decide)

end Task1Formalization