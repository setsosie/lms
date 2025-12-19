/-
  FROM-SCRATCH CATEGORY THEORY: Yoneda Lemma

  Building on the verified HomFunctor structure.
  Key insight: Need separate universe params for Obj and Hom levels.
-/
import Mathlib.Tactic.Common
import Mathlib.Logic.Basic

namespace LMS.Yoneda

universe u v u' v'

-- Category structure with two universe parameters
structure Category where
  Obj : Type u
  Hom : Obj → Obj → Type v
  id : (x : Obj) → Hom x x
  comp : {x y z : Obj} → Hom x y → Hom y z → Hom x z
  id_comp : ∀ {x y : Obj} (f : Hom x y), comp (id x) f = f
  comp_id : ∀ {x y : Obj} (f : Hom x y), comp f (id y) = f
  assoc : ∀ {w x y z : Obj} (f : Hom w x) (g : Hom x y) (h : Hom y z),
          comp (comp f g) h = comp f (comp g h)

-- Functor structure
structure Func (C : Category.{u, v}) (D : Category.{u', v'}) where
  obj : C.Obj → D.Obj
  map : {x y : C.Obj} → C.Hom x y → D.Hom (obj x) (obj y)
  map_id : ∀ (x : C.Obj), map (C.id x) = D.id (obj x)
  map_comp : ∀ {x y z : C.Obj} (f : C.Hom x y) (g : C.Hom y z),
             map (C.comp f g) = D.comp (map f) (map g)

-- Natural transformation with ext attribute
@[ext]
structure NatTrans {C : Category.{u, v}} {D : Category.{u', v'}} (F G : Func C D) where
  app : (x : C.Obj) → D.Hom (F.obj x) (G.obj x)
  naturality : ∀ {x y : C.Obj} (f : C.Hom x y),
               D.comp (F.map f) (app y) = D.comp (app x) (G.map f)

-- Opposite category wrapper
structure Op (α : Type u) where
  unop : α

def op {α : Type u} (x : α) : Op α := ⟨x⟩

-- Opposite category (same universe levels, just swapped morphisms)
def OpCat (C : Category.{u, v}) : Category.{u, v} where
  Obj := Op C.Obj
  Hom := fun x y => C.Hom y.unop x.unop  -- reversed
  id := fun x => C.id x.unop
  comp := fun f g => C.comp g f  -- reversed composition
  id_comp := fun f => C.comp_id f
  comp_id := fun f => C.id_comp f
  assoc := fun f g h => (C.assoc h g f).symm

-- Category of types at universe v
-- Objects: Type v (lives in Type (v+1))
-- Morphisms: A → B (lives in Type v)
def TypeCat : Category.{v+1, v} where
  Obj := Type v
  Hom := fun A B => A → B
  id := fun _ => fun x => x
  comp := fun f g => fun x => g (f x)
  id_comp := fun _ => rfl
  comp_id := fun _ => rfl
  assoc := fun _ _ _ => rfl

-- The Hom functor h_x = Hom(-, x) : C^op → TypeCat
def HomFun (C : Category.{v+1, v}) (x : C.Obj) : Func (OpCat C) TypeCat.{v} where
  obj := fun y => C.Hom y.unop x
  map := fun {y y'} f => fun g => C.comp f g
  map_id := fun y => by
    funext g
    exact C.id_comp g
  map_comp := fun {y y' y''} f f' => by
    funext g
    exact C.assoc f' f g

-- Yoneda forward map: η ↦ η.app (op x) (C.id x)
def yonedaForward {C : Category.{v+1, v}} {x : C.Obj}
    {F : Func (OpCat C) TypeCat.{v}} (η : NatTrans (HomFun C x) F) : F.obj (op x) :=
  η.app (op x) (C.id x)

-- Yoneda backward map: a ↦ natural transformation
def yonedaBackward {C : Category.{v+1, v}} {x : C.Obj}
    {F : Func (OpCat C) TypeCat.{v}} (a : F.obj (op x)) : NatTrans (HomFun C x) F where
  app := fun y => fun f => (F.map f) a
  naturality := fun {y₁ y₂} g => by
    funext f
    show F.map (C.comp g f) a = (F.map g) ((F.map f) a)
    have h : C.comp g f = (OpCat C).comp f g := rfl
    rw [h]
    have hmc := F.map_comp f g
    rw [hmc]
    rfl

-- Yoneda lemma: the maps are mutually inverse
-- Forward ∘ Backward = id
theorem yoneda_right_inv {C : Category.{v+1, v}} {x : C.Obj}
    {F : Func (OpCat C) TypeCat.{v}} (a : F.obj (op x)) :
    yonedaForward (yonedaBackward a) = a := by
  simp only [yonedaForward, yonedaBackward]
  -- Goal: F.map (C.id x) a = a
  have h : C.id x = (OpCat C).id (op x) := rfl
  have hid := F.map_id (op x)
  -- hid : F.map ((OpCat C).id (op x)) = TypeCat.id (F.obj (op x))
  -- TypeCat.id = fun x => x, so this is just "id"
  calc F.map (C.id x) a = F.map ((OpCat C).id (op x)) a := by rw [h]
    _ = (TypeCat.id (F.obj (op x))) a := by rw [hid]
    _ = a := rfl

-- Backward ∘ Forward = id
theorem yoneda_left_inv {C : Category.{v+1, v}} {x : C.Obj}
    {F : Func (OpCat C) TypeCat.{v}} (η : NatTrans (HomFun C x) F) :
    yonedaBackward (yonedaForward η) = η := by
  ext y
  simp only [yonedaBackward, yonedaForward]
  -- Goal: (fun f => F.map f (η.app (op x) (C.id x))) = η.app y
  funext f
  -- Goal: F.map f (η.app (op x) (C.id x)) = η.app y f
  -- By naturality at f : (OpCat C).Hom (op x) y = C.Hom y.unop x
  have nat := η.naturality f
  -- nat : TypeCat.comp ((HomFun C x).map f) (η.app y) = TypeCat.comp (η.app (op x)) (F.map f)
  -- LHS = fun g => η.app y ((HomFun C x).map f g) = fun g => η.app y (C.comp f g)
  -- RHS = fun g => (F.map f) (η.app (op x) g)
  have key := congr_fun nat (C.id x)
  -- key : η.app y ((HomFun C x).map f (C.id x)) = (F.map f) (η.app (op x) (C.id x))
  -- (HomFun C x).map f (C.id x) = C.comp f (C.id x) = f
  simp only [HomFun, TypeCat] at key
  rw [C.comp_id] at key
  exact key.symm

-- The Yoneda Lemma: Nat(Hom(-, x), F) ≃ F(x)
#check @yoneda_right_inv
#check @yoneda_left_inv

end LMS.Yoneda
