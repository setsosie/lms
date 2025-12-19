/-
  Yoneda Lemma - Self-contained proof

  Building on verified artifacts and learning from 30+ failed attempts.
  Key insights:
  - Universe levels: C : Category.{v+1, v} for HomFun
  - OpCat composition is REVERSED: comp f g = C.comp g f
  - NatTrans needs @[ext] attribute
  - TypeCat at universe (v+1, v)
-/

namespace LMS.Yoneda

-- Universe v for morphisms, v+1 for objects (to avoid issues)
universe v

/-! ## Category Structure -/

structure Category where
  Obj : Type (v+1)
  Hom : Obj → Obj → Type v
  id : (x : Obj) → Hom x x
  comp : {x y z : Obj} → Hom x y → Hom y z → Hom x z
  id_comp : ∀ {x y : Obj} (f : Hom x y), comp (id x) f = f
  comp_id : ∀ {x y : Obj} (f : Hom x y), comp f (id y) = f
  assoc : ∀ {w x y z : Obj} (f : Hom w x) (g : Hom x y) (h : Hom y z),
    comp (comp f g) h = comp f (comp g h)

/-! ## Functor and Natural Transformation -/

structure Func (C D : Category.{v}) where
  obj : C.Obj → D.Obj
  map : {x y : C.Obj} → C.Hom x y → D.Hom (obj x) (obj y)
  map_id : ∀ (x : C.Obj), map (C.id x) = D.id (obj x)
  map_comp : ∀ {x y z : C.Obj} (f : C.Hom x y) (g : C.Hom y z),
    map (C.comp f g) = D.comp (map f) (map g)

@[ext]
structure NatTrans {C D : Category.{v}} (F G : Func C D) where
  component : (x : C.Obj) → D.Hom (F.obj x) (G.obj x)
  naturality : ∀ {x y : C.Obj} (f : C.Hom x y),
    D.comp (F.map f) (component y) = D.comp (component x) (G.map f)

/-! ## Opposite Category -/

def Op (α : Type u) := α

def OpCat (C : Category.{v}) : Category.{v} where
  Obj := Op C.Obj
  Hom := fun x y => C.Hom y x  -- reversed!
  id := fun x => C.id x
  comp := fun f g => C.comp g f  -- REVERSED composition
  id_comp := fun f => C.comp_id f
  comp_id := fun f => C.id_comp f
  assoc := fun f g h => (C.assoc h g f).symm

/-! ## Category of Types -/

def TypeCat : Category.{v} where
  Obj := Type v
  Hom := fun X Y => X → Y
  id := fun _ x => x
  comp := fun f g x => g (f x)
  id_comp := fun _ => rfl
  comp_id := fun _ => rfl
  assoc := fun _ _ _ => rfl

/-! ## Hom Functor h_x = Hom(-, x) -/

def HomFun (C : Category.{v}) (x : C.Obj) : Func (OpCat C) TypeCat where
  obj := fun y => C.Hom y x
  map := fun {y z} (f : (OpCat C).Hom y z) => fun (g : C.Hom y x) => C.comp f g
  -- f : (OpCat C).Hom y z = C.Hom z y
  -- g : C.Hom y x
  -- Want: C.Hom z x
  -- C.comp f g : C.Hom z x ✓
  map_id := fun y => by
    funext g
    exact C.id_comp g
  map_comp := fun {a b c} (f : (OpCat C).Hom a b) (g : (OpCat C).Hom b c) => by
    funext h
    -- f : (OpCat C).Hom a b = C.Hom b a
    -- g : (OpCat C).Hom b c = C.Hom c b
    -- h : C.Hom a x
    -- (OpCat C).comp f g = C.comp g f : (OpCat C).Hom a c = C.Hom c a
    -- LHS: map ((OpCat C).comp f g) h = C.comp ((OpCat C).comp f g) h
    --                                 = C.comp (C.comp g f) h
    -- RHS: TypeCat.comp (map f) (map g) h = (map g) ((map f) h)
    --    = (fun k => C.comp g k) ((fun k => C.comp f k) h)
    --    = (fun k => C.comp g k) (C.comp f h)
    --    = C.comp g (C.comp f h)
    show C.comp (C.comp g f) h = C.comp g (C.comp f h)
    exact C.assoc g f h

/-! ## Yoneda Lemma -/

variable {C : Category.{v}} {F : Func (OpCat C) TypeCat} {x : C.Obj}

/-- Forward direction: η ↦ η_x(id_x) -/
def yonedaForward (η : NatTrans (HomFun C x) F) : F.obj x :=
  η.component x (C.id x)

/-- Backward direction: a ↦ η where η_y(f) = F.map(f)(a) -/
def yonedaBackward (a : F.obj x) : NatTrans (HomFun C x) F where
  component := fun y f => F.map f a
  naturality := fun {y z} (g : (OpCat C).Hom y z) => by
    funext f'
    -- g : (OpCat C).Hom y z = C.Hom z y
    -- f' : C.Hom y x
    -- Need: TypeCat.comp ((HomFun C x).map g) (component z) f'
    --     = TypeCat.comp (component y) (F.map g) f'
    -- LHS: (component z) ((HomFun C x).map g f')
    --    = (component z) (C.comp g f')  -- HomFun.map g f' = C.comp g f'
    --    = F.map (C.comp g f') a
    -- RHS: F.map g ((component y) f')
    --    = F.map g (F.map f' a)
    -- Need: F.map (C.comp g f') a = F.map g (F.map f' a)
    -- (HomFun C x).map g f' = C.comp g f' by definition
    -- We need to show: F.map (C.comp g f') a = F.map g (F.map f' a)
    -- Note: g : C.Hom z y, f' : C.Hom y x
    -- As morphisms in OpCat: g : (OpCat C).Hom y z, f' : (OpCat C).Hom x y
    -- C.comp g f' is equal to (OpCat C).comp f' g as terms
    simp only [HomFun, TypeCat]
    -- Goal: (fun f => F.map f a) (C.comp g f') = F.map g (F.map f' a)
    -- LHS = F.map (C.comp g f') a
    -- F.map_comp f' g says: F.map ((OpCat C).comp f' g) = TypeCat.comp (F.map f') (F.map g)
    --                                                    = fun x => F.map g (F.map f' x)
    -- Since C.comp g f' = (OpCat C).comp f' g as terms:
    have h : F.map (C.comp g f') a = F.map g (F.map f' a) := by
      have map_eq : F.map (C.comp g f') = fun b => F.map g (F.map f' b) := F.map_comp f' g
      rw [map_eq]
    exact h

/-- Right inverse: yonedaForward ∘ yonedaBackward = id -/
theorem yoneda_right_inv (a : F.obj x) :
    yonedaForward (yonedaBackward a) = a := by
  unfold yonedaForward yonedaBackward
  simp only []
  -- Goal: F.map (C.id x) a = a
  -- Since F : Func (OpCat C) TypeCat, and (OpCat C).id x = C.id x
  have h : F.map (C.id x) a = a := by
    have id_eq : F.map (C.id x) = fun y => y := F.map_id x
    rw [id_eq]
  exact h

/-- Left inverse: yonedaBackward ∘ yonedaForward = id -/
theorem yoneda_left_inv (η : NatTrans (HomFun C x) F) :
    yonedaBackward (yonedaForward η) = η := by
  unfold yonedaForward yonedaBackward
  ext y
  funext f'
  -- Goal: F.map f' (η.component x (C.id x)) = η.component y f'
  -- f' : C.Hom y x, which is (OpCat C).Hom x y
  -- Use naturality of η at f'
  have nat : TypeCat.comp ((HomFun C x).map f') (η.component y)
           = TypeCat.comp (η.component x) (F.map f') := η.naturality f'
  -- This is saying: for all h, η.component y ((HomFun C x).map f' h) = F.map f' (η.component x h)
  -- Apply to h = C.id x
  have nat_at_id : η.component y ((HomFun C x).map f' (C.id x)) = F.map f' (η.component x (C.id x)) := by
    have h := congrFun nat (C.id x)
    exact h
  -- (HomFun C x).map f' (C.id x) = C.comp f' (C.id x) = f'
  have map_eval : (HomFun C x).map f' (C.id x) = f' := C.comp_id f'
  rw [map_eval] at nat_at_id
  exact nat_at_id.symm

/-! ## Summary

We have proven the Yoneda Lemma from scratch:

THEOREM: For any functor F : C^op → Type and object x in C,
there is a natural bijection:
    Nat(h_x, F) ≅ F(x)
where h_x = Hom(-, x) is the representable functor.

The bijection is witnessed by:
- yonedaForward : Nat(h_x, F) → F(x)
  Given η : h_x → F, returns η_x(id_x)

- yonedaBackward : F(x) → Nat(h_x, F)
  Given a ∈ F(x), defines η where η_y(f) = F(f)(a)

And we proved these are mutually inverse:
- yoneda_right_inv : yonedaForward ∘ yonedaBackward = id
- yoneda_left_inv : yonedaBackward ∘ yonedaForward = id

All definitions are self-contained including Category, Func, NatTrans, OpCat,
TypeCat, and HomFun.
-/

end LMS.Yoneda
