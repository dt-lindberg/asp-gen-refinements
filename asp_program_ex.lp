% ------------------------------------------------------------
%  1.  CONSTANT DECLARATIONS   (UTF‑8 encoded strings)
% ------------------------------------------------------------

% Employees
employee("Bonita"; "Yvette"; "Tabitha").

% Prices (shared between the two sub‑problems)
price(225; 275; 325).          % for the employee‑wood problem
price(150; 160; 170; 180).     % for the masseuse‑client problem

% Wood types
wood_type("ash"; "poplar"; "sandalwood").

% Clients
client("Aimee"; "Ginger"; "Freda"; "Hannah").

% Masseuses
masseuse("Lynda"; "Nancy"; "Teri"; "Whitney").

% ------------------------------------------------------------
%  2.  RELATION:  match/3   (employee ↔ price ↔ wood_type)
% ------------------------------------------------------------

% --- generate the search space --------------------------------
% For every employee E we choose **exactly one** pair (P,W).
% The pair must come from the respective domains.
{ match(E,P,W) : price(P), wood_type(W) } = 1 :- employee(E).

% --- optional functional constraints ----------------------------
% (a) each price can be used by at most one employee
:- price(P), 2 { match(E,P,_) : employee(E) }.

% (b) each wood type can be used by at most one employee
:- wood_type(W), 2 { match(E,_,W) : employee(E) }.

% ------------------------------------------------------------
%  3.  RELATION:  schedule/3   (masseuse ↔ client ↔ price)
% ------------------------------------------------------------

% --- generate the search space --------------------------------
% For every client C we choose **exactly one** pair (M,P).
{ schedule(M,C,P) : masseuse(M), price(P) } = 1 :- client(C).

% --- optional functional constraints ----------------------------
% (a) each price can be used by at most one client
:- price(P), 2 { schedule(_,C,P) : client(C) }.

% (b) each masseuse can treat at most one client
:- masseuse(M), 2 { schedule(M,_,P) : price(P) }.

% ------------------------------------------------------------
%  4.  (Optional)  Example of a classic “queen‑placement’’ rule
% ------------------------------------------------------------
% The line below is only illustrative – it shows how a binary
% predicate with two arguments would be declared in the same program.
% assign(Ir,Ic) :- row(Ir), col(Ic), not conflict(Ir,Ic).

% ------------------------------------------------------------
%  5.  Queries / integrity checks (you can comment them out)
% ------------------------------------------------------------

% Show all generated match/3 atoms
#show match/3.
% Show all generated schedule/3 atoms
#show schedule/3.

% clingo your_program.lp 0   % ← command‑line call, kept as a comment

% ----------------------------------------------------------------------
%  All‑different constraints (no option in any category will ever be used more than once)
% ----------------------------------------------------------------------
% each client appears once
:- schedule(M1,C,_), schedule(M2,C,_), M1 != M2.

% each price appears once
:- schedule(M1,_,P), schedule(M2,_,P), M1 != M2.

% each masseuse appears once (implicit by definition, but kept for symmetry)
:- schedule(M,_,_), schedule(M2,_,_), M != M2.

% ----------------------------------------------------------------------
%  Clue 1: Hannah paid more than Teri's client.
% ----------------------------------------------------------------------
:- schedule(M_H, "Hannah", P_H),
   schedule("Teri", C_T, P_T),
   P_H <= P_T.

% ----------------------------------------------------------------------
%  Clue 2: Freda paid 20 dollars more than Lynda's client.
% ----------------------------------------------------------------------
:- schedule(M_F, "Freda", P_F),
   schedule("Lynda", C_L, P_L),
   P_F != P_L + 20.

% ----------------------------------------------------------------------
%  Clue 3: Hannah paid 10 dollars less than Nancy's client.
% ----------------------------------------------------------------------
:- schedule(M_H, "Hannah", P_H),
   schedule("Nancy", C_N, P_N),
   P_H != P_N - 10.

% ----------------------------------------------------------------------
%  Clue 4: Nancy's client, Hannah and Ginger were all different clients.
% ----------------------------------------------------------------------
% (the constants "Hannah" and "Ginger" are themselves different, so no extra rule needed)
:- schedule("Nancy", "Hannah", _).   % Nancy cannot have Hannah as client
:- schedule("Nancy", "Ginger", _).   % Nancy cannot have Ginger as client

% ----------------------------------------------------------------------
%  Clue 5: Hannah was either the person who paid $180 or Lynda's client.
% ----------------------------------------------------------------------
% at least one of the two conditions must hold
:- schedule(M_H, "Hannah", P_H),
   schedule("Lynda", C_L, _),
   P_H != 180,
   C_L != "Hannah".

% ----------------------------------------------------------------------
%  Domain facts (optional, for completeness – can be omitted if the solver already knows the domains)
% ----------------------------------------------------------------------
price(150). price(160). price(170). price(180).
client("Aimee"). client("Ginger"). client("Freda"). client("Hannah").
masseuse("Lynda"). masseuse("Nancy"). masseuse("Teri"). masseuse("Whitney").

% ----------------------------------------------------------------------
%  End of constraints
% ----------------------------------------------------------------------
