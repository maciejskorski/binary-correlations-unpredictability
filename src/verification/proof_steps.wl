(* Symbolic check of the key proof steps (Wolfram Language). Blocks z in {+-1}^n, z_1 = y, n = m+1; ones are -1. *)
chi[B_, z_] := Times @@ z[[B]];  tup[n_] := Tuples[{1, -1}, n];  allm = ConstantArray[-1, #] &;

(* 1. Walsh identity: sum_B (-1)^|B| chi_B(z) = prod_j (1 - z_j) *)
c1 = AllTrue[Range[6], Function[n, With[{z = Array[s, n]}, Expand[Sum[(-1)^Length[B] chi[B, z], {B, Subsets[Range[n]]}]] === Expand[Product[1 - z[[j]], {j, n}]]]]];

(* 2. Extremal law: b_B = (-1)^(|B|-1) gam, equal prefix/suffix marginals, advantage 2^(m-1) gam/(1-(2^m-1) gam) *)
c2 = AllTrue[Range[3], Function[m, Module[{n = m + 1, p, g},
  p[z_] := 2^-n (1 + gam - gam Product[1 - z[[j]], {j, n}]);
  g[w_] := (p[Prepend[w, 1]] - p[Prepend[w, -1]])/(p[Prepend[w, 1]] + p[Prepend[w, -1]]);
  AllTrue[DeleteCases[Subsets[Range[n]], {}], Simplify[Sum[p[z] chi[#, z], {z, tup[n]}] - (-1)^(Length[#] - 1) gam] === 0 &] &&
  AllTrue[tup[m], Simplify[Sum[p[Append[#, t]] - p[Prepend[#, t]], {t, {1, -1}}]] === 0 &] &&
  Simplify[g[allm[m]]/2 - 2^(m - 1) gam/(1 - (2^m - 1) gam)] === 0]]];

(* 3. Auxiliary law q, every B containing position 1 and every sign sg, arbitrary g:
      equal marginals and E_q[y g(w)] = sg*btilde/k exactly *)
c3 = AllTrue[Range[3], Function[m, Module[{n = m + 1, gg = Array[gf, 2^m], ok = True, W = tup[m]},
  Do[Module[{d = Max[B] - 1, k, A, bt, qq},
    k = n - d; A = DeleteCases[B, 1] - 1;
    bt = 2^-m Sum[gg[[i]] chi[A, W[[i]]], {i, 2^m}];
    qq[z_] := 2^-n (1 + sg/k Sum[chi[B + j, z], {j, 0, k - 1}]);
    ok = ok && Simplify[Sum[qq[z] z[[1]] gg[[FirstPosition[W, Rest[z]][[1]]]], {z, tup[n]}] - sg bt/k] === 0 &&
      AllTrue[W, Expand[Sum[qq[Append[#, t]] - qq[Prepend[#, t]], {t, {1, -1}}]] === 0 &]],
    {B, Select[Subsets[Range[n]], MemberQ[#, 1] &]}]; ok]]];

(* 4. Walsh inversion of the context law (basis of the total-variation step) *)
c4 = AllTrue[Range[3], Function[m, Module[{n = m + 1, p = Array[pf, 2^(m + 1)], Z = tup[m + 1]},
  AllTrue[tup[m], Expand[Sum[p[[i]] If[Rest[Z[[i]]] === #, 1, 0], {i, 2^n}] - 2^-m Sum[Sum[p[[i]] chi[A, Rest[Z[[i]]]], {i, 2^n}] chi[A, #], {A, Subsets[Range[m]]}]] === 0 &]]]];

(* 5. Trap source: every |b_B| = gam, P(w_-) = 2^-m (1+(2^m-1) gam), g(w_-) = -2^m gam/(1+(2^m-1) gam),
      and log2(1+|g|)/gam -> 2^m/ln 2 *)
c5 = AllTrue[Range[3], Function[m, Module[{n = m + 1, p},
  p[z_] := If[z === allm[n], (1 + (2^n - 1) gam)/2^n, (1 - gam)/2^n];
  AllTrue[DeleteCases[Subsets[Range[n]], {}], Simplify[Sum[p[z] chi[#, z], {z, tup[n]}] - (-1)^Length[#] gam] === 0 &] &&
  Simplify[p[Prepend[allm[m], 1]] + p[allm[n]] - 2^-m (1 + (2^m - 1) gam)] === 0 &&
  Simplify[(p[Prepend[allm[m], 1]] - p[allm[n]])/(p[Prepend[allm[m], 1]] + p[allm[n]]) + 2^m gam/(1 + (2^m - 1) gam)] === 0 &&
  Limit[Log[1 + 2^m gam/(1 + (2^m - 1) gam)]/(gam Log[2]), gam -> 0] === 2^m/Log[2]]]];

(* 6. Runs-of-ones potential inequality, symbolic in l = leading ones: s*G(w) <= 1 + phi(w) - phi(w'),
      equality for s=+1 and for s=-1 with l<m; at l=m the slack is 2^(m+1) *)
G[l_] := 2^(l + 1) - 1;  phi[l_] := 2^(l + 1) - 2;
c6 = {Simplify[G[l] - (1 + phi[l] - phi[0])] === 0, Simplify[-G[l] - (1 + phi[l] - phi[l + 1])] === 0, Simplify[-G[m] - (1 + phi[m] - phi[m])] === -2^(m + 1)};

(* 7. Runs-of-ones source: stationary law exactly in theta; {zeroth, first} order of max_B |b_B| is {0, m+1} *)
c7 = Table[Module[{n = m + 1, W = tup[m], T, pi, g, p, ell, Z = tup[m + 1], bs},
  ell[w_] := LengthWhile[w, # == -1 &];  g[w_] := th (2^(ell[w] + 1) - 1);
  T = ConstantArray[0, {2^m, 2^m}];
  Do[Do[T[[i, FirstPosition[W, Prepend[Most[W[[i]]], y]][[1]]]] += (1 + y g[W[[i]]])/2, {y, {1, -1}}], {i, 2^m}];
  pi = Normalize[First[NullSpace[Transpose[T] - IdentityMatrix[2^m]]], Total];
  p[z_] := pi[[FirstPosition[W, Rest[z]][[1]]]] (1 + z[[1]] g[Rest[z]])/2;
  bs = Table[Normal[Series[Sum[p[z] chi[B, z], {z, Z}], {th, 0, 1}]], {B, DeleteCases[Subsets[Range[n]], {}]}];
  {Max[Abs[Coefficient[#, th, 0] & /@ bs]], Max[Abs[Coefficient[#, th, 1] & /@ bs]]}], {m, 1, 3}];

(* 8. XOR-route loss 2 Sqrt[2^(m+1)-1] ~ 2^((m+3)/2); noisy-parity H_min formula; margin = 1 at N ~ 1.15e6 *)
c8a = Limit[2 Sqrt[2^(m + 1) - 1]/2^((m + 3)/2), m -> Infinity] === 1;
c8b = Chop[N[(-Log[2, 2^-m (1/2 + dl)^(N0 - m)] - (m + (N0 - m) (1 - Log[2, 1 + 2 dl])) /. {m -> 2, N0 -> 10, dl -> 1/10})]] === 0;
delta = (2^(1/100) - 1)/2;
c8c = n /. FindRoot[2 delta n/Sqrt[4 n Log[n]] == 1, {n, 10^6}];

{c1, c2, c3, c4, c5, c6, c7, c8a, c8b, c8c}
