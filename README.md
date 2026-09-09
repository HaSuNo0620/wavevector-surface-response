# wavevector-surface-response

空間的に周期的な外場を受ける一成分液体–気体界面について、Monte Carlo 計算から**界面自由エネルギーの波数応答**と、その内部に含まれる界面変位・厚み揺らぎ・microscopic packing の寄与を解析するための再現可能な計算パイプラインである。

## 研究の中心量

外場を

$$
V_{\mathrm{ext}}(\mathbf r)
=
A\cos(\mathbf q\cdot\mathbf r)
$$

とし、外場振幅 $A$ に対する界面自由エネルギー密度を

$$
\gamma(A,\mathbf q)
=
\gamma_0
+
\frac12\chi_\gamma(\mathbf q)A^2
+
O(A^4)
$$

と展開する。

ここで

$$
\chi_\gamma(\mathbf q)
=
\left.
\frac{\partial^2\gamma}{\partial A^2}
\right|_{A=0}
$$

を、本研究では **表面張力の波数応答**、より厳密には **wavevector-resolved susceptibility of interfacial free energy** と呼ぶ。

Hamiltonian を

$$
H(A)=H_0+A X_{\mathbf q}
$$

と書けば、平衡統計力学から

$$
\frac{\partial F}{\partial A}
=
\langle X_{\mathbf q}\rangle_A
$$

および

$$
\left.
\frac{\partial^2F}{\partial A^2}
\right|_{A=0}
=
-\beta\operatorname{Var}(X_{\mathbf q})
$$

が成り立つ。

したがって本研究の主目的は、自由な毛細管波そのものを測ることではなく、**外場に共役な密度応答が波数とともにどのように変化するか**を調べることである。

## 研究目的

中心となる問いは、外場に対する界面自由エネルギー応答が

$$
q\ll q_*
$$

の長波長領域から

$$
q\sim q_*
$$

の分子 packing 領域へ移るとき、その内部構造がどのように変化するか、である。

ここで $q_*$ は bulk liquid structure factor $S(q)$ の第一ピーク位置を表す。

密度揺らぎを概念的に

$$
\delta\rho_q(z)
=
\delta\rho_q^{\mathrm{int}}(z)
+
\delta\rho_q^{\mathrm{pack}}(z)
$$

と分ける。前者には界面位置・界面形状・二界面間の collective motion が含まれ、後者にはそれでは説明できない microscopic density fluctuation が含まれる。

外場に共役な観測量については

$$
X_q
=
X_{\mathrm{int}}(q)
+
X_{\mathrm{pack}}(q)
$$

と書き、その分散を

$$
\operatorname{Var}(X_q)
=
\operatorname{Var}(X_{\mathrm{int}})
+
2\operatorname{Cov}(X_{\mathrm{int}},X_{\mathrm{pack}})
+
\operatorname{Var}(X_{\mathrm{pack}})
$$

と分解する。

したがって、本研究で主要に見るのは

$$
\chi_{\mathrm{total}}(q)
=
\chi_{\mathrm{int}}(q)
+
2\chi_{\mathrm{cross}}(q)
+
\chi_{\mathrm{pack}}(q)
$$

という **外場応答の sector decomposition** である。

## 毛細管波理論の位置づけ

界面変位が十分長波長で単独自由界面として記述できる場合には

$$
\delta\rho_q(z)
\simeq
-h_q\rho_0'(z)
$$

と書ける。

さらに標準的な capillary-wave Hamiltonian

$$
\mathcal H_{\mathrm{CW}}
=
\frac{\gamma A_\parallel}{2}
\sum_q q^2|h_q|^2
$$

が成立するなら

$$
\langle |h_q|^2\rangle
\propto
\frac{1}{q^2}
$$

が期待される。

ただし、**この $q^{-2}$ 則は本研究の合格条件ではない。**

今回の外場は全密度に作用し、界面高さへ直接結合する外場ではない。また周期境界下の液体–気体 slab には二つの界面が存在するため、単一自由界面の capillary-wave 理論がそのまま成立するとは限らない。

したがって $q^{-2}$ 則は、低波数側で界面 sector が単純な Goldstone / capillary 極限へ接続するかを調べる**補助的な漸近診断**として扱う。

## 二界面系の collective coordinate

周期境界下の slab には二つの界面があるため、それぞれの変位を $h_1(q)$、$h_2(q)$ とする。

そこから

$$
H_q
=
\frac{h_1(q)+h_2(q)}{2}
$$

を slab 全体の**並進モード**、

$$
W_q
=
h_2(q)-h_1(q)
$$

を slab の**厚み・breathing モード**と定義する。

現在の pilot 計算では、単純な並進 $H_q$ よりも厚み揺らぎ $W_q$ の方が大きく、二界面の collective motion を理解する上で重要であることが示唆されている。

## 密度相関と固有モード解析

面内波数 $q_\parallel$ ごとに

$$
\rho_q(z,t)
$$

を保存し、

$$
\delta\rho_q(z,t)
=
\rho_q(z,t)
-
\langle\rho_q(z)\rangle
$$

から共分散核

$$
C(z,z';q_\parallel)
=
\left\langle
\delta\rho_q(z)
\delta\rho_q^*(z')
\right\rangle
$$

を構成する。

これを

$$
\int dz'\,
C(z,z';q_\parallel)v_n(z')
=
\lambda_n(q_\parallel)v_n(z)
$$

として固有値分解する。

この固有モード解析は、外場応答そのものではなく、密度揺らぎがどのような $z$ 方向構造を持つかを理解するための補助解析である。

二界面の局所変位 template が張る部分空間と、共分散固有モード群との principal angle も測定する。ただし、capillary subspace が主要固有空間に強く現れない場合でも、外場に共役な observable への寄与が小さいとは限らない。

## 現在までの pilot で見えていること

現時点の結果は次のように整理できる。

1. 低波数では、外場に共役な observable の sector decomposition において interface-like 成分が大きい。
2. $q$ を増やすと microscopic / packing sector の寄与が増大する。
3. $q\sim q_*$ 付近では packing sector が支配的になる。
4. 二界面系では slab 並進 $H_q$ より thickness / breathing mode $W_q$ の方が強い。
5. 単純な $\langle|h_q|^2\rangle\propto q^{-2}$ は、現在の有限サイズ pilot では明瞭には確認されていない。
6. このこと自体は外場応答の描像を否定しない。毛細管波則は本研究の主目的ではなく、低 $q$ 側の補助診断である。

したがって、現在の中心仮説は単純な

$$
\text{Goldstone mode}
\longrightarrow
\text{packing mode}
$$

という固有モード交差ではなく、

$$
\boxed{
\text{外場応答の内部構造が}
\quad
\text{interface-like}
\longrightarrow
\text{packing-like}
\quad
\text{へ移る}
}
$$

という crossover である。

## 第一段階：$q_\perp=0$

現在は

$$
V_{\mathrm{ext}}(x,z)
=
A\cos(q_\parallel x+q_\perp z)
$$

に対して

$$
q_\perp=0
$$

に集中している。

この場合、外場は

$$
V_{\mathrm{ext}}(x)
=
A\cos(q_\parallel x)
$$

となり、界面の絶対 $z$ 位置に対する余分な位相感度を避けられる。

ただし $q_\perp=0$ でも有限 $q_\parallel$ の界面変位との結合が消えるわけではない。

## PBC と波数

現在の lateral mode は

$$
q_\parallel
=
\frac{2\pi n_x}{L_x}
$$

を満たす離散波数を使用する。

傾斜外場を扱う場合には、さらに

$$
q_\perp L_z
=
2\pi n_z
$$

を満たす必要がある。

これは slab 系と bulk reference 系の PBC 非整合による偽の線形応答を避けるために重要である。

## 計算パイプライン

1. NVT Lennard-Jones 液体–気体 slab を Metropolis Monte Carlo 法で生成する。
2. PBC と整合する波数 $q_\parallel$ に対して $\rho_q(z,t)$ を保存する。
3. 外場に共役な密度モードの分散を求める。
4. 界面 sector と microscopic packing sector に分解する。
5. 二界面の collective coordinate $H_q$、$W_q$ を解析する。
6. 密度共分散の固有モード・principal angle・residual PC を補助的に解析する。
7. bulk reference を用いて界面過剰 susceptibility $\chi_\gamma(q)$ を構成する。
8. 必要に応じて小振幅外場 $\pm A$ の直接計算と fluctuation–dissipation relation を比較する。
9. box size、run length、replica、自己相関時間に対する収束性を検証する。

## 主なスクリプト

```text
scripts/run_mc.py
scripts/analyze_modes.py
scripts/analyze_capillary_coordinates.py
scripts/analyze_translation_thickness.py
scripts/analyze_thickness_packing_coupling.py
scripts/fit_capillary_massive_model.py
scripts/plot_mode_summary.py
```

## 現在の pilot 条件

full-box pilot では

$$
T^*=1.0,
\qquad
L_x=L_y=16,
\qquad
L_z=48,
\qquad
N=4800
$$

を用いている。

面内波数は

$$
q_\parallel
=
\frac{2\pi n_x}{16}
$$

として、低波数域から packing scale 付近まで複数の $n_x$ を同一 trajectory から計測する。

これらの値は現時点では physics pilot 用であり、publication-quality の最終条件ではない。

## 注意

interface / packing 分解や $H_q$、$W_q$ は、密度揺らぎの内部構造を理解するための診断量であり、定義や射影 convention に依存する。

一方、最終的な熱力学的観測量は

$$
\chi_\gamma(\mathbf q)
=
\left.
\frac{\partial^2\gamma}{\partial A^2}
\right|_{A=0}
$$

である。

したがって本研究では、**毛細管波理論を先に仮定してデータを解釈するのではなく、外場応答を第一に測定し、その内部構造として界面変位・厚み揺らぎ・packing を分解する**という順序を採用する。
