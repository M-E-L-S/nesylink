/-!
  这是Task3的针对性lean建模和策略证明。
  对应task3 - 状态机/BFS搜索agent的实现。
  完整包含了Task3涉及到的环境建模，证明了安全性、可达性、BFS完备性。
-/

namespace Task3Formalization

/-!
  基本环境建模
-/
abbrev Position := Nat × Nat

inductive Action where
  | wait | up | down | left | right | attack | interact | shield
  deriving DecidableEq, Repr

inductive Room where
  | startRoom | monsterHall | keyRoom
  deriving DecidableEq, Repr

structure SymbolicState where
  room : Room
  player : Position
  monsters : List (Room × Position)
  chests : List (Room × Position)
  health : Nat
  keys : Nat
  deriving DecidableEq, Repr

def manhattan (a b : Position) : Nat :=
  let dx := if a.1 ≤ b.1 then b.1 - a.1 else a.1 - b.1
  let dy := if a.2 ≤ b.2 then b.2 - a.2 else a.2 - b.2
  dx + dy

def adjacent (a b : Position) : Prop := manhattan a b = 1

def inBounds (p : Position) : Prop := p.1 < 10 ∧ p.2 < 8

def applyMove (r : Room) (p : Position) (a : Action) : Room × Position :=
-- 特殊的房间切换机制
  match r, a with
  | Room.startRoom, Action.left =>
      if p.1 == 0 then (Room.monsterHall, (8, 4)) else (r, (p.1 - 1, p.2))
  | Room.startRoom, Action.right =>
      (r, (if p.1 == 9 then 9 else p.1 + 1, p.2))
  | Room.monsterHall, Action.left =>
      if p.1 == 0 then (Room.keyRoom, (8, 4)) else (r, (p.1 - 1, p.2))
  | Room.monsterHall, Action.right =>
      if p.1 == 9 then (Room.startRoom, (1, 4)) else (r, (p.1 + 1, p.2))
  | Room.keyRoom, Action.left =>
      (r, (if p.1 == 0 then 0 else p.1 - 1, p.2))
  | Room.keyRoom, Action.right =>
      if p.1 == 9 then (Room.monsterHall, (1, 4)) else (r, (p.1 + 1, p.2))
  | _, Action.up => (r, (p.1, if p.2 == 0 then 0 else p.2 - 1))
  | _, Action.down => (r, (p.1, if p.2 == 7 then 7 else p.2 + 1))
  | _, _ => (r, p)

def isSafeTile (p : Position) : Prop := inBounds p

def adjacentMonster (s : SymbolicState) : Prop :=
  ∃ mPos, (s.room, mPos) ∈ s.monsters ∧ adjacent s.player mPos

def adjacentChest (s : SymbolicState) : Prop :=
  ∃ cPos, (s.room, cPos) ∈ s.chests ∧ adjacent s.player cPos

def canAttack (s : SymbolicState) (m : Room × Position) : Prop :=
  m ∈ s.monsters ∧ m.1 = s.room ∧ adjacent s.player m.2 ∧ s.health > 1

def canOpenChest (s : SymbolicState) (c : Room × Position) : Prop :=
  c ∈ s.chests ∧ c.1 = s.room ∧ adjacent s.player c.2

inductive Step : SymbolicState → Action → SymbolicState → Prop where
  | move
      {s : SymbolicState} {a : Action} :
      a ∈ [Action.up, Action.down, Action.left, Action.right] →
      isSafeTile (applyMove s.room s.player a).2 →
      Step s a { s with room := (applyMove s.room s.player a).1, player := (applyMove s.room s.player a).2 }
  | moveBlocked
      {s : SymbolicState} {a : Action} :
      a ∈ [Action.up, Action.down, Action.left, Action.right] →
      ¬ isSafeTile (applyMove s.room s.player a).2 →
      Step s a s
  | attackMonster
      {s : SymbolicState} {m : Room × Position} :
      canAttack s m →
      Step s Action.attack { s with monsters := s.monsters.erase m }
  | openChest
      {s : SymbolicState} {c : Room × Position} :
      ¬ adjacentMonster s →
      canOpenChest s c →
      Step s Action.interact { s with chests := s.chests.erase c, keys := s.keys + 1 }
  | actionNoEffect
      {s : SymbolicState} {a : Action} :
      a ∈ [Action.attack, Action.interact] →
      ¬ adjacentMonster s →
      ¬ adjacentChest s →
      Step s a s
  | wait {s : SymbolicState} : Step s Action.wait s
  | shield {s : SymbolicState} : Step s Action.shield s

inductive Exec : SymbolicState → List Action → SymbolicState → Prop where
  | nil {s : SymbolicState} : Exec s [] s
  | cons {s t u : SymbolicState} {a : Action} {rest : List Action} :
      Step s a t → Exec t rest u → Exec s (a :: rest) u

def GoalReached (s : SymbolicState) : Prop :=
  s.monsters = [] ∧ s.keys > 0 ∧ s.room = Room.startRoom ∧ s.player.1 = 9

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
    (hsafe : isSafeTile (applyMove s.room s.player a).2) :
    SafeState t := by
  cases h with
  | move _ hsafe' => exact hsafe'
  | moveBlocked _ hblocked => exact False.elim (hblocked hsafe)
  | attackMonster => cases ha <;> contradiction
  | openChest => cases ha <;> contradiction
  | actionNoEffect => cases a <;> simp_all
  | wait => cases ha <;> contradiction
  | shield => cases ha <;> contradiction


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

def roomEqB (r1 r2 : Room) : Bool :=
  match r1, r2 with
  | Room.startRoom, Room.startRoom => true
  | Room.monsterHall, Room.monsterHall => true
  | Room.keyRoom, Room.keyRoom => true
  | _, _ => false

def posEqB (p1 p2 : Position) : Bool := p1.1 == p2.1 && p1.2 == p2.2
def targetEqB (t1 t2 : Room × Position) : Bool := roomEqB t1.1 t2.1 && posEqB t1.2 t2.2
def isSafeTileB (p : Position) : Bool := decide (p.1 < 10) && decide (p.2 < 8)
def adjacentB (p1 p2 : Position) : Bool := manhattan p1 p2 == 1

def findTarget? (xs : List (Room × Position)) (p : Room × Position → Bool) : Option (Room × Position) :=
  match xs with
  | [] => none
  | x :: rest => if p x then some x else findTarget? rest p

def canAttackB (s : SymbolicState) (m : Room × Position) : Bool :=
  roomEqB m.1 s.room && adjacentB s.player m.2 && decide (s.health > 1)

def canOpenChestB (s : SymbolicState) (c : Room × Position) : Bool :=
  roomEqB c.1 s.room && adjacentB s.player c.2

def symbolicStepFn (s : SymbolicState) : Action → SymbolicState
  | Action.up =>
      let rp := applyMove s.room s.player Action.up
      if isSafeTileB rp.2 then { s with room := rp.1, player := rp.2 } else s
  | Action.down =>
      let rp := applyMove s.room s.player Action.down
      if isSafeTileB rp.2 then { s with room := rp.1, player := rp.2 } else s
  | Action.left =>
      let rp := applyMove s.room s.player Action.left
      if isSafeTileB rp.2 then { s with room := rp.1, player := rp.2 } else s
  | Action.right =>
      let rp := applyMove s.room s.player Action.right
      if isSafeTileB rp.2 then { s with room := rp.1, player := rp.2 } else s
  | Action.attack =>
      match findTarget? s.monsters (canAttackB s) with
      | some m => { s with monsters := s.monsters.erase m }
      | none => s
  | Action.interact =>
      match findTarget? s.chests (canOpenChestB s) with
      | some c => { s with chests := s.chests.erase c, keys := s.keys + 1 }
      | none => s
  | _ => s

def runPlanFn : SymbolicState → List Action → SymbolicState
  | s, [] => s
  | s, a :: rest => runPlanFn (symbolicStepFn s a) rest

def allActions : List Action :=
  [Action.wait, Action.up, Action.down, Action.left, Action.right, Action.attack, Action.interact, Action.shield]

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
          have hStepMem : symbolicStepFn s head ∈ expandFrontier frontier :=
            mem_expandFrontier_of_mem head hs
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

theorem step_move_compute {s : SymbolicState} {a : Action} {room' : Room} {pos' : Position}
    (ha : a ∈ [Action.up, Action.down, Action.left, Action.right])
    (hcomp : applyMove s.room s.player a = (room', pos'))
    (hsafe : pos'.1 < 10 ∧ pos'.2 < 8) :
    Step s a { s with room := room', player := pos' } := by
  have h_base : Step s a { s with room := (applyMove s.room s.player a).1, player := (applyMove s.room s.player a).2 } := by
    apply Step.move ha
    unfold isSafeTile inBounds
    rw [hcomp]
    exact hsafe
  rw [hcomp] at h_base
  exact h_base

theorem task3_completable_if_subplans_exist
    {init nearMonster afterKill nearChest afterChest final : SymbolicState}
    {toMonster toChest toExit : List Action}
    {monster chest : Room × Position}
    (hToMonster : Exec init toMonster nearMonster)
    (hAttackable : canAttack nearMonster monster)
    (hAfterKill : afterKill = { nearMonster with monsters := nearMonster.monsters.erase monster })
    (hToChest : Exec afterKill toChest nearChest)
    (hNoAdjacentMonster : ¬ adjacentMonster nearChest)
    (hOpenable : canOpenChest nearChest chest)
    (hAfterChest : afterChest = { nearChest with chests := nearChest.chests.erase chest, keys := nearChest.keys + 1 })
    (hToExit : Exec afterChest toExit final)
    (hFinalMonsters : final.monsters = [])
    (hFinalKeys : final.keys > 0)
    (hFinalAtExit : final.room = Room.startRoom ∧ final.player.1 = 9) :
    TaskCompletable init := by
  have hAttackStep : Step nearMonster Action.attack afterKill := by
    rw [hAfterKill]; exact Step.attackMonster hAttackable
  have hOpenStep : Step nearChest Action.interact afterChest := by
    rw [hAfterChest]; exact Step.openChest hNoAdjacentMonster hOpenable
  have hPhase1 : Exec init (toMonster ++ [Action.attack]) afterKill :=
    exec_append hToMonster (Exec.cons hAttackStep Exec.nil)
  have hPhase2 : Exec init ((toMonster ++ [Action.attack]) ++ toChest) nearChest :=
    exec_append hPhase1 hToChest
  have hPhase3 : Exec init (((toMonster ++ [Action.attack]) ++ toChest) ++ [Action.interact]) afterChest :=
    exec_append hPhase2 (Exec.cons hOpenStep Exec.nil)
  have hAll : Exec init ((((toMonster ++ [Action.attack]) ++ toChest) ++ [Action.interact]) ++ toExit) final :=
    exec_append hPhase3 hToExit
  exact ⟨_, final, hAll, ⟨hFinalMonsters, hFinalKeys, hFinalAtExit⟩⟩

def t3Init : SymbolicState :=
  { room := Room.startRoom, player := (4, 4), monsters := [(Room.monsterHall, (5, 3))],
    chests := [(Room.keyRoom, (5, 4))], health := 5, keys := 0 }

def t3Monster : Room × Position := (Room.monsterHall, (5, 3))
def t3Chest : Room × Position := (Room.keyRoom, (5, 4))

def t3ToMonster : List Action :=
  [Action.left, Action.left, Action.left, Action.left, Action.left, Action.left, Action.left, Action.left]

def t3ToChest : List Action :=
  [Action.left, Action.left, Action.left, Action.left, Action.left, Action.left, Action.left, Action.left]

def t3ToExit : List Action :=
  [Action.right, Action.right, Action.right, Action.right, Action.right, Action.right, Action.right,
   Action.right, Action.right, Action.right, Action.right, Action.right, Action.right, Action.right,
   Action.right, Action.right, Action.right, Action.right, Action.right, Action.right, Action.right]

def t3NearMonster : SymbolicState := { t3Init with room := Room.monsterHall, player := (5, 4) }
def t3AfterKill : SymbolicState := { t3NearMonster with monsters := [] }
def t3NearChest : SymbolicState := { t3AfterKill with room := Room.keyRoom, player := (6, 4) }
def t3AfterChest : SymbolicState := { t3NearChest with chests := [], keys := 1 }
def t3Final : SymbolicState := { t3AfterChest with room := Room.startRoom, player := (9, 4) }

theorem t3_exec_to_monster : Exec t3Init t3ToMonster t3NearMonster := by
  unfold t3ToMonster
  iterate 8 (apply Exec.cons; exact step_move_compute (by decide) rfl (by decide))
  exact Exec.nil

theorem t3_exec_to_chest : Exec t3AfterKill t3ToChest t3NearChest := by
  unfold t3ToChest
  iterate 8 (apply Exec.cons; exact step_move_compute (by decide) rfl (by decide))
  exact Exec.nil

theorem t3_exec_to_exit : Exec t3AfterChest t3ToExit t3Final := by
  unfold t3ToExit
  iterate 21 (apply Exec.cons; exact step_move_compute (by decide) rfl (by decide))
  exact Exec.nil

theorem task3_concrete_completable : TaskCompletable t3Init := by
  exact task3_completable_if_subplans_exist
    (init := t3Init)
    (nearMonster := t3NearMonster)
    (afterKill := t3AfterKill)
    (nearChest := t3NearChest)
    (afterChest := t3AfterChest)
    (final := t3Final)
    (toMonster := t3ToMonster)
    (toChest := t3ToChest)
    (toExit := t3ToExit)
    (monster := t3Monster)
    (chest := t3Chest)
    t3_exec_to_monster
    (by unfold canAttack t3NearMonster t3Init t3Monster adjacent manhattan; decide)
    rfl
    t3_exec_to_chest
    (by intro h; rcases h with ⟨m, hm, _⟩; simp [t3NearChest, t3AfterKill, t3NearMonster, t3Init] at hm)
    (by unfold canOpenChest t3NearChest t3AfterKill t3NearMonster t3Init t3Chest adjacent manhattan; decide)
    rfl
    t3_exec_to_exit
    (by decide)
    (by decide)
    (by decide)

end Task3Formalization
