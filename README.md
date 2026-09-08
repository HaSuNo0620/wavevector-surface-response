# wavevector-surface-response

空間的に周期的な外場を受ける一成分液体–気体界面について、Monte Carlo 計算から波数分解された界面自由エネルギー応答と密度揺らぎモードを解析するための再現可能な計算パイプラインである。

本研究で主に扱う応答量は、外場振幅 $A$ に対する界面自由エネルギー密度（表面張力）の二次応答

$$
\gamma(A,\mathbf q)
=
\gamma_0
+
\frac{1}{2}\chi_\gamma(\mathbf q)A^2
+
O(A^4),
$$

$$
\chi_\gamma(\mathbf q)
=
\left.
\frac{\partial^2\gamma}{\partial A^2}
\right|_{A=0}
$$

である。

ここではこれを **表面張力の波数応答**、あるいは **wavevector-resolved susceptibility of interfacial free energy** とみなす。

## 研究目的

第一段階の目的は、液体–気体界面の密度揺らぎが、長波長では界面変位に対応する Goldstone / capillary モードとして振る舞い、分子スケールでは microscopic packing モードへと混成していく可能性を検証することである。

密度揺らぎを

$$
\delta\rho_q(z)
=
h_q\,\phi_G(z)
+
\delta\rho_q^{\perp}(z),
$$

と分解する。ここで並進 Goldstone モードの基準関数を

$$
\phi_G(z)
\equiv
-\frac{d\rho_0(z)}{dz}
$$

と定義する。

したがって、より具体的には

$$
\delta\rho_q(z)
=
h_q\left[-\rho_0'(z)\right]
+
\delta\rho_q^{\perp}(z)
$$

である。

外場に共役な揺らぎ応答については、全応答を概念的に

$$
\chi_{\mathrm{total}}
=
\chi_{hh}
+
2\chi_{h\perp}
+
\chi_{\perp\perp}
$$

と分解し、界面変位セクター、混合セクター、packing セクターの寄与を調べる。

## 密度相関と固有モード解析

面内波数 $q_\parallel$ に対して密度モード $\rho_q(z,t)$ を計測し、その揺らぎ

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

この行列を

$$
\int dz'\,
C(z,z';q_\parallel)v_n(z')
=
\lambda_n(q_\parallel)v_n(z)
$$

として固有値分解する。

離散化した数値計算では通常の Hermitian 固有値問題

$$
C\mathbf v_n
=
\lambda_n\mathbf v_n
$$

として解く。

各固有モードが並進 Goldstone モードにどれだけ近いかを

$$
O_n(q)
=
\frac{
\left|
\langle v_n,-\rho_0'\rangle
\right|^2
}{
\langle v_n,v_n\rangle
\langle\rho_0',\rho_0'\rangle
}
$$

で評価する。

長波長 capillary 領域では、理想的には Goldstone-like branch に対して

$$
O_G(q)\simeq 1
$$

かつ

$$
\lambda_G(q)\propto \frac{1}{q^2}
$$

が期待される。したがって

$$
q^2\lambda_G(q)
$$

が低波数域でほぼ一定になるかを主要な診断量の一つとする。

## 計算パイプライン

1. NVT Lennard-Jones 液体–気体 slab を Metropolis Monte Carlo 法で生成する。
2. PBC と整合する面内波数 $q_\parallel$ に対して $\rho_q(z,t)$ を保存する。
3. $C(z,z';q_\parallel)$ を構築する。
4. 共分散行列を固有値分解し、$-\rho_0'(z)$ との overlap を計算する。
5. 各 snapshot を Goldstone 射影成分とその直交成分へ分解する。
6. 各 sector の揺らぎ寄与と closure error を計算する。
7. box size、sampling 長、replica、自己相関時間に対する収束性を検証する。
8. 必要に応じて小振幅外場 $\pm A$ を加え、平衡揺らぎ応答と有限差分自由エネルギー曲率を比較する。

## 現在の第一段階

最初の解析では $q_\perp=0$ に集中する。

外場を

$$
V_{\mathrm{ext}}(x,z)
=
A\cos(q_\parallel x+q_\perp z)
$$

とすると、$q_\perp\neq0$ では界面の絶対位置に対する位相感度が追加される。一方、$q_\perp=0$ ではこの絶対位相問題を避けつつ、有限 $q_\parallel$ の界面変位モードと microscopic packing の混成を調べることができる。

なお、$q_\perp=0$ であっても有限 $q_\parallel$ の capillary mode との結合そのものが消えるわけではない。

## Goldstone-to-packing crossover の判定

本研究で検証したい中心仮説は

$$
\text{Goldstone / capillary}
\quad\longrightarrow\quad
\text{mode mixing}
\quad\longrightarrow\quad
\text{microscopic packing}
$$

という波数に沿った crossover である。

これを支持する結果として、少なくとも次を期待する。

- 低波数で Goldstone-like eigenmode の overlap $O_G(q)$ が大きい。
- 低波数で $q^2\lambda_G(q)$ が概ね一定となる。
- 分子スケールの波数に近づくにつれて $O_G(q)$ が低下する。
- 同時に非並進モードとの混成が増える。
- packing sector および cross sector の寄与が増大する。

液体構造の代表的な microscopic scale として、bulk liquid structure factor $S(q)$ の第一ピーク位置 $q_*$ を基準に用いる。

## 応答の closure

射影後の観測量を

$$
X
=
X_h+X_\perp
$$

と書けば、分散は

$$
\operatorname{Var}(X)
=
\operatorname{Var}(X_h)
+
2\operatorname{Cov}(X_h,X_\perp)
+
\operatorname{Var}(X_\perp)
$$

を満たす。

コードではこの恒等式の残差を closure error として保存し、射影・規格化・実装の整合性を確認する。

## PBC と波数

現在の lateral mode は box 長 $L_x$ に対して

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

これは過去の解析で問題となった、slab 系と bulk reference 系の PBC 非整合による偽の線形応答を避けるために重要である。

## リポジトリ構成

```text
config/                  Monte Carlo・解析条件
src/wvsr/                MC および解析ライブラリ
scripts/                 実行用スクリプト
results/                 計算結果
.github/workflows/       GitHub Actions による自動テスト・pilot 計算
```

主なスクリプトは以下である。

```text
scripts/run_mc.py
scripts/analyze_modes.py
scripts/plot_mode_summary.py
```

## 現在の pilot 計算

現在の full-box pilot では

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

この pilot は Goldstone-to-packing crossover の有無を予備判定するためのものであり、現時点では publication-quality の収束を保証するものではない。

## 注意

Goldstone / packing 分解の各成分は、射影の定義に依存する診断量である。一方で全界面自由エネルギー応答 $\chi_\gamma$ は熱力学的観測量である。

したがって最終的には、モード分解だけでなく

$$
\chi_\gamma(\mathbf q)
=
\left.
\frac{\partial^2\gamma}{\partial A^2}
\right|_{A=0}
$$

そのものとの対応を検証することを目標とする。
