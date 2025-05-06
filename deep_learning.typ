#import "report_template.typ": *
#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1": plot
#import "@preview/fletcher:0.5.7": *
#import "@preview/wrap-it:0.1.1": wrap-content

#let get_hr = context({
  let selector = heading.where(level: 2).before(here())
  let headings = query(selector)
  if headings.len() > 0 {
    headings.last().body
  }
})

#let batch = $abs(b)$

#show: init.with(
  header-left: "DeepLearning 入門",
  header-right: [#get_hr],
)

#set page(
  margin: (
    top: 2.8cm,
    bottom: 1.8cm,
    x: 1.5cm,
  )
)

#let subtoi_format = (n,m) => {
    set text(size: 12pt, weight: 500)
    [（#h(1pt)#num2alphabet(m)#h(1pt)）]
}

#show: topbar.with(
  titlefield: [
    #set text(size: 35pt, weight: 500);
    #h(-3pt);
    DeepLearning入門\
    #v(-10mm)
    #set text(size: 12pt, weight: 500);
    〜『ゼロから作るDeep Learning』を読んで〜
  ],
  parameters: (
  ),
)

#v(-1.2cm)
#h(1fr)#text(size: 14pt)[著：博ノ助]

#outline()

#pagebreak()

= 本書で用いる表現について
- ベクトルは太文字の小文字で表す。（$bold(x)$）
- 行列は太文字の大文字で表す。（$bold(X)$）
- スカラーは通常の小文字で表す。（$x$）
- 行列の要素は定義のうえ、通常の小文字と添字で表す。添字は行、列の順に表記し、わかりにくい場合を除き区切り文字をつけない。（$x_(i j)$$x_(i+1,j)$）
- 行列の要素について断りがない時、行列名$bold(X)$を用いて$[bold(X)]_(i j)$と表すことがある。添字は同様。
- $N$次元ベクトルは大きさ$1 times N$の行列とみなし、いわゆる行ベクトルとする。
- 列ベクトルの要素表示は通常、行ベクトルと転置記号$top$を用いて表す。わかりやすさのため、列ベクトルそのものを書くこともある。

#pagebreak()

= 多層パーセプトロン（MLP）
この章では、DeepLearningの基礎である多層パーセプトロン（MLP）について学ぶ。最小単位ニューロンの構造を紹介し、その具体的な利用法と、限界について触れることでのちの章で出現する様々な要素を導入するモチベーションを与える。\
== ニューロンの始まり
ニューロンは生物の神経細胞を模した情報処理の単位である。入力信号を受け取り、それに基づいて出力信号を生成する。

#align(center)[
  #diagram(
    node-stroke: black,
    edge-corner-radius: none,
    node-inset: 0pt,
    spacing: (80pt, 0pt),
    node-shape: "circle",
    // Nodes
    node((0,0), [$hat(f)_?$], name: <10>, width: 1.0cm),
    edge((-1,0), (0,0), "-|>", label: [$x$]),
    edge((0,0), (1,0), "-|>", label: [$hat(f)_?(x)$]),
  )
]


この際、入力の強さによって信号を出力するかどうかを決めることにしてみよう。これは生物的に行われていることであるが、これを数学的に表現するとステップ関数を用いることができる。
#table(columns: (2fr, 1fr),align: center+horizon,stroke: none)[
$
  f_text("step")(x) = cases(
    thick 1 & quad text("if") quad x >= 0,
    thick 0 & quad text("otherwise"),
  )
$
][
#cetz.canvas({
  plot.plot(
    size: (4, 1.8),
    axis-style: "school-book",
    x-min: -2,
    x-max: 2,
    y-min: -0.2,
    y-max: 1.2,
    x-tick-step: none,
    y-tick-step: 1,
    x-ticks: (),
    {
      plot.add(
        domain: (-3, 3),
        samples: 1000,
        x => if x < 0 { 0 } else { 1 },
        style: (stroke: 2pt + blue)
      )
    }
  )
})
]
この関数では入力$x$が$0$以上で発火することになる。しかしこれでは、$0$か$1$しか伝播しないため、帰納的にすべてのニューロンが発火してしまう。そこで、ニューロン同士の結合の強さ$w$と、発火する閾値を調整する$b$というパラメータを導入することでニューロンの結合は意味を持つ。また、複数の入力に対してもその結合を考慮することができるように、ニューロンの出力は以下のように定義される。
$
  hat(f)_text("step")(bold(x)) =colon f_text("step")(sum_i w_i x_i+b) = cases(
    thick 1 & quad text("if") quad sum_i w_i x_i >= -b,
    thick 0 & quad text("otherwise"),
  )
$

$w_i$は$i$番目の入力$x_i$に対する結合強度を表す。このようにしてニューロンは複数の入力を受け取り,それらの結合強度と閾値を考慮して出力を決定することができる。これをパーセプトロンという。このパーセプトロンによる実例を見てみよう。

#align(center)[
#diagram(
  node-stroke: black,
  edge-corner-radius: none,
  node-inset: 0pt,
  spacing: (40pt, 10pt),
  node-shape: "circle",

  // Nodes
  node((0,0), [$x$], name: <x>, width: 1.0cm, stroke: none),
  node((0,2), [$y$], name: <y>, width: 1.0cm, stroke: none),

  node((1,1), [$hat(f)_text("step")$], name: <and>, width: 1.0cm),
  node((1.4,2), [$b=-1.5$], name: <label>, width: 2.0cm, stroke: none),
  node((2,1), [$z$], name: <z>, width: 1.0cm, stroke: none),

  edge(<x>, <and>, "-|>", label: [$w_1=1$]),
  edge(<y>, <and>, "-|>", label: [$w_2=1$], label-anchor: "north", label-sep: -0.12cm),
  edge(<and>, <z>, "-|>"),
)
]
$
  z = f_text("step")(x + y - 1.5) = cases(
    thick 1 & quad text("if") quad x + y >= 1.5,
    thick 0 & quad text("otherwise"),
  )
$

上のパーセプトロンはANDゲートを模している。$0$か$1$しかとらない入力$x$と$y$に対して、ともに$1$のときのみ出力$z$が$1$になり、それ以外のときは$0$になる。このことは上の式からわかるだろう。

次にORゲートを模したパーセプトロンを見てみよう。
#align(center)[
#diagram(
  node-stroke: black,
  edge-corner-radius: none,
  node-inset: 0pt,
  spacing: (40pt, 10pt),
  node-shape: "circle",

  // Nodes
  node((0,0), [$x$], name: <x>, width: 1.0cm, stroke: none),
  node((0,2), [$y$], name: <y>, width: 1.0cm, stroke: none),

  node((1,1), [$hat(f)_text("step")$], name: <or>, width: 1.0cm),
  node((1.4,2), [$b=-0.5$], name: <label>, width: 2.0cm, stroke: none),
  node((2,1), [$z$], name: <z>, width: 1.0cm, stroke: none),

  edge(<x>, <or>, "-|>", label: [$w_1=1$]),
  edge(<y>, <or>, "-|>", label: [$w_2=1$], label-anchor: "north", label-sep: -0.12cm),
  edge(<or>, <z>, "-|>"),
)
]

上のパーセプトロンはORゲートを模している。$0$か$1$しかとらない入力$x$と$y$に対して、少なくとも片方が$1$のときのみ出力$z$が$1$になり、それ以外のときは$0$になる。同じように入力$x$に対して$0$ならば$1$、$1$ならば$0$を出力するNOTゲートを模したパーセプトロンを実装できる。
#align(center)[
#diagram(
  node-stroke: black,
  edge-corner-radius: none,
  node-inset: 0pt,
  spacing: (40pt, 10pt),
  node-shape: "circle",

  // Nodes
  node((0,1), [$x$], name: <x>, width: 1.0cm, stroke: none),

  node((1,1), [$hat(f)_text("step")$], name: <not>, width: 1.0cm),
  node((1.4,2), [$b=-0.5$], name: <label>, width: 2.0cm, stroke: none),
  node((2,1), [$z$], name: <z>, width: 1.0cm, stroke: none),

  edge(<x>, <not>, "-|>", label: [$w_1=-1$]),
  edge(<not>, <z>, "-|>"),
)
]

== 多層パーセプトロン

では,XORゲートはどうだろうか。XORゲートは$0$か$1$しかとらない入力$x$と$y$に対して、片方が$1$のときのみ出力$z$が$1$になり、それ以外のときは$0$になる関数である。しかし、XORゲートは上のようなパーセプトロンでは実装できない。右下のようなグラフを書いてみる。

#let figure = align(center)[
  #cetz.canvas({
    plot.plot(
      size: (2.8, 2.8),
      axis-style: "school-book",
      x-min: -0.2,
      x-max: 1.2,
      y-min: -0.2,
      y-max: 1.2,
      x-tick-step: 1,
      y-tick-step: 1,
      {
        plot.add(
          ((0,0), (1,1)),
          mark: "o",
          style: (stroke: none),
        )
        plot.add(
          ((1,0), (0,1)),
          mark: "o",
          style: (stroke: none),
        )
      }
    )
  })
]
#wrap-content(
  figure,
  column-gutter: 1em,
  row-gutter: 0em,
  align: bottom + right,
)[
二つの入力を$x$軸$y$軸として、出力を色分けした。青色は$0$、赤色は$1$を表している。ニューロンは
$
  hat(f)_text("step")(w_1 x + w_2 y + b) = cases(
    thick 1 & quad text("if") quad w_1 x + w_2 y >= -b,
    thick 0 & quad text("otherwise"),
  )
$
と表されるのであるが、これは直線$w_1 x + w_2 y + b=0$を境界に$x y$平面を分けることを意味する。
]

しかし、グラフを見ればわかるように直線1つで青と赤の点を分けることができない。これを、線形分離不可能という。XORゲートは線形分離不可能な関数である。
したがって、XORゲートは1つのニューロンでは実装できない。しかし、XORゲートはいくつかの他の論理ゲートを組み合わせることで実装できることが知られている。


#align(center)[
#diagram(
  node-stroke: black,
  edge-corner-radius: none,
  node-inset: 0pt,
  spacing: (40pt, 15pt),
  node-shape: "circle",

  // Nodes
  node((0,1), [$x$], name: <x>, width: 1.0cm, stroke: none),
  node((0,3), [$y$], name: <y>, width: 1.0cm, stroke: none),

  node((1.35,1), [OR], name: <or>, width: 1.0cm),
  node((1,3), [AND], name: <and>, width: 1.0cm),
  node((1.7,3), [NOT], name: <not>, width: 1.0cm),
  node((2.5,2), [AND], name: <and1>, width: 1.0cm),
  node((3.5,2), [$z$], name: <z>, width: 1.0cm, stroke: none),

  edge(<x>, <or>, "-|>"),
  edge(<y>, <or>, "-|>"),
  edge(<x>, <and>, "-|>"),
  edge(<y>, <and>, "-|>"),
  edge(<and>, <not>, "-|>"),
  edge(<or>, <and1>, "-|>"),
  edge(<not>, <and1>, "-|>"),
  edge(<and1>, <z>, "-|>"),
)
]

このように、XORゲートはORゲートとANDゲートとNOTゲートを組み合わせることで実装できる。複数のパーセプトロンをまるで層を重ねるように組み合わせることで、より複雑な関数を実装することができる。これを多層パーセプトロン（MLP）とよぶ。データの入力を受け付ける層を入力層、出力を表す層を出力層とよぶ。その間にある層を隠れ層とよぶ。入力層と出力層をみればデータの入出力の形だけはわかる。\

== 活性化関数
多層パーセプトロンは、複数のニューロンを組み合わせることでより複雑な関数を実装することができる。しかし、上のようなステップ関数$f_text("step")$では、出力が離散的で値の表現が乏しい。そのため、連続的な値を出力する関数を用いる。これら、ニューロンの出力を決定する関数を活性化関数とよぶ。\

また、のちに多層パーセプトロンはいわゆる深層学習に使われるが、その際、微分が大きな役割を果たす。そのため、微分可能であるような関数が現在の活性化関数として用いられている。活性化関数の進化によって、ニューロンの「発火」という性質は薄れていった。一方で、パラメータ$w, b$は意味を変えつつもそのまま残っており、重み及びバイアスとよばれるようになった。

活性化関数には様々な種類があるが、ここでは代表的なものを紹介する。\
=== シグモイド関数
#table(columns: (2fr, 1fr),align: center+horizon,stroke: none)[
$
  f_text("sigmoid")(x) = frac(1, 1 + exp(-x))
$
][
#cetz.canvas({
  plot.plot(
    size: (3.5, 2.5),
    axis-style: "school-book",
    x-min: -8,
    x-max: 8,
    y-min: -0.2,
    y-max: 1.2,
    x-tick-step: none,
    y-tick-step: 0.5,
    x-ticks: (),
    {
      plot.add(
        domain: (-9, 9),
        samples: 1000,
        x => 1 / (1 + calc.exp(-x)),
        style: (stroke: 2pt + blue)
      )
    }
  )
})
]
シグモイド関数は、入力$x$に対して$0$から$1$の値を出力する微分可能な関数である。これは、ニューロンの出力を確率として解釈することができる。ステップ関数と形が似ているため、ステップ関数の滑らかな近似としてよく用いられていた。しかし、入力の絶対値が十分大きい場合に、微分係数が$0$に近くなってしまうため、学習が進まなくなるという問題#footnote[これを勾配消失問題という。]がある。\

=== ReLU関数
#table(columns: (2fr, 1fr),align: center+horizon,stroke: none)[
$
  f_text("ReLU")(x) = cases(
    thick x & quad text("if") quad x >= 0,
    thick 0 & quad text("otherwise"),
  ) quad = max(0, x)
$
][
#cetz.canvas({
  plot.plot(
    size: (3.5, 2.5),
    axis-style: "school-book",
    x-min: -5,
    x-max: 5,
    y-min: -1,
    y-max: 5.2,
    x-tick-step: none,
    y-tick-step: none,
    x-ticks: (),
    {
      plot.add(
        domain: (-9, 9),
        samples: 1000,
        x => if x < 0 { 0 } else { x },
        style: (stroke: 2pt + blue)
      )
    }
  )
})
]
ReLU関数は、入力$x$に対して$0$以上の値を出力する微分可能な関数である。これは、入力が$0$以上のときはそのまま出力し、$0$未満のときは$0$を出力する。シグモイド関数と比べて、勾配消失問題が起こりにくいという利点があるため、現在ではデフォルトで用いられている。面白いことに、ReLU関数はステップ関数を積分したものになっている。

=== Softmax関数
$
  f_text("softmax")(x_i) = frac(exp(x_i), sum_j exp(x_j))
$

Softmax関数は、他の活性化関数とは異なり入力全体のベクトル$bold(x)$を用いる.$i$番目の要素$x_i$に対して、$x_i$の全体に占める割合に似たものを出力する.出力は$0$以上の値を持ち、全体の和は$1$になる。確率分布を表すために用いられる。

Softmax関数を実装する際の注意点として、入力$x_i$の値が大きいときに、$exp(x_i)$がオーバーフローしてしまうことがある。これを防ぐために、全体の最大値を引いてから計算する。値が小さくても$exp(x_i)$は$0$に近づくだけなので問題ない。入力を定数分だけ引いても出力は変わらないので、この操作で出力は変化しない。

$
  &C = text("const.")&\
  &frac(exp(x_i - C), sum_j exp(x_j - C))= frac(e^(-C) exp(x_i), sum_j e^(-C)exp(x_j))\
  & wide =frac(e^(-C) exp(x_i), e^(-C) sum_j exp(x_j))\
  & wide =frac(exp(x_i), sum_j exp(x_j))
$

== 行列表現とAffine層
ここでは、層構造をもつ多層パーセプトロンについて、効率的な計算をおこなうために行列計算に帰着できることを示す。\

=== Affine層
Affine層は、入力$x$に対して、重み$W$とバイアス$b$を用いて出力$y$を計算する層である。活性化関数については一度目を瞑ることにする。最も基本的な形は以下の図で示されるようなものである。

#align(center)[
  #set text(size: 9pt, weight: 500)
  #diagram(
  node-stroke: black,
  edge-corner-radius: none,
  node-inset: 0pt,
  spacing: (10pt, 24pt),
  node-shape: "circle",

  // Nodes
  node((0,0), [$1$], name: <10>, width: 1.0cm),
  node((0,1), [$2$], name: <11>, width: 1.0cm),
  node((0,2), [$3$], name: <12>, width: 1.0cm),
  node((0,3), [$dots.v$], name: <13>, width: 1.0cm, stroke: none),
  node((0,4), [$n-2$], name: <14>, width: 1.0cm),
  node((0,5), [$n-1$], name: <15>, width: 1.0cm),
  node((0,6), [$n$], name: <16>, width: 1.0cm),

  node((10,1), [$1$], name: <21>, width: 1.0cm),
  node((10,2), [$2$], name: <22>, width: 1.0cm),
  node((10,3), [$dots.v$], name: <23>, width: 1.0cm, stroke: none),
  node((10,4), [$m-1$], name: <24>, width: 1.0cm),
  node((10,5), [$m$], name: <25>, width: 1.0cm),

  edge(<10>, <21>),
  edge(<11>, <21>),
  edge(<12>, <21>),
  edge(<14>, <21>),
  edge(<15>, <21>),
  edge(<16>, <21>),
  edge(<10>, <22>),
  edge(<11>, <22>),
  edge(<12>, <22>),
  edge(<14>, <22>),
  edge(<15>, <22>),
  edge(<16>, <22>),
  edge(<10>, <24>),
  edge(<11>, <24>),
  edge(<12>, <24>),
  edge(<14>, <24>),
  edge(<15>, <24>),
  edge(<16>, <24>),
  edge(<10>, <25>),
  edge(<11>, <25>),
  edge(<12>, <25>),
  edge(<14>, <25>),
  edge(<15>, <25>),
  edge(<16>, <25>),
  )
]

いくつかのニューロンが並列に配置されており、これを層あるいはレイヤという。レイヤ内のニューロンの数を層の大きさとよび、上図ではサイズ$n$のレイヤとサイズ$m$のレイヤが結合されている。また、上図では省略されているが、各入力は個別の重みとバイアスを持つ。

注意すべきなのが、各ニューロンの各入力について重みがあるという点である。例えば、右側レイヤの1番目のニューロンは左側レイヤの1番目のニューロンからの入力に対して重み$w_(1 arrow.l 1)$を持つ。加えて、右側レイヤの2番目のニューロンは左側レイヤの1番目のニューロンからの入力に先ほどとは異なる重み$w_(2 arrow.l 1)$を持つのである。ただし、バイアスはニューロンごとに1つしか持たない。\

今現在、右側レイヤに注目する。左側レイヤの$i$番目のニューロンの出力を$x_i$、右側レイヤの$j$番目がもつ$x_i$に対する重みを$w_(j arrow.l i)$と書くことにする。右側レイヤの$j$番目のニューロンのバイアスを$b_j$、出力を$y_j$とすると、以下のように表すことができる。
$
  y_j = sum_(1<= i <= n) w_(j arrow.l i) x_i + b_j
$
ここで、以下のような重み行列$bold(W)$とバイアスベクトル$bold(b)$を定義する。
$
  bold(W) colon&= mat(
    w_(1 arrow.l 1), w_(2 arrow.l 1), dots, w_(m arrow.l 1);
    w_(1 arrow.l 2), w_(2 arrow.l 2), dots, w_(m arrow.l 2);
    dots.v, dots.v, dots.down, dots.v;
    w_(1 arrow.l n), w_(2 arrow.l n), dots, w_(m arrow.l n);
  ) quad in MM^(n times m)\
  bold(b) colon&= mat(b_1, b_2, dots, b_m;)
$
$bold(W)$の$i,j$成分$w_(i j)$は$w_(j arrow.l i)$に一致している.この行列とベクトルを用いると、左側レイヤの出力$bold(x)$に対して、右側レイヤの出力$bold(y)$は以下のように表すことができる。
$
  bold(y) = bold(x) bold(W) + bold(b)
$
簡単な行列計算で済むのである.

=== バッチ付きAffine層
機械学習では複数のデータを同時に扱うことで計算を効率化することが一般的である。ここでいう、データとは$bold(x)$や$bold(y)$を1単位としている。時に、何億というデータを扱うこともあるため、コンピュータの性能が許す限り、一度に多くのデータを扱いたいのである。これをバッチ処理とよび、いくつかのデータをまとめたものをバッチとよぶ。また、一度に扱うデータの数をバッチサイズとよび、$batch in NN$と書くことにする。\

Affine層では、単に入出力をベクトルから行列に拡張するだけでバッチ処理を実現できる。次の$bold(X), bold(Y)$に入出力を改める。またバイアスも$bold(B)$に改める。ただし、バッチ内の$i$番目のデータを$bold(x)_i, bold(y)_i$とする。\
$
  bold(X) colon&= mat(
    bold(x)_1, bold(x)_2, dots, bold(x)_batch;
  )^top quad &in &MM^(batch times n)\
  bold(Y) colon&= mat(
    bold(y)_1, bold(y)_2, dots, bold(y)_batch;
  )^top quad &in &MM^(batch times m)\
  bold(B) colon&= mat(
    bold(b), bold(b), dots, bold(b);
  )^top quad &in &MM^(batch times m)
$
これは先ほどと同様に行列の積で計算できる。
$
  bold(Y) = bold(X) dot bold(W) + bold(b)
$
これは以下のように正当化される。
$
  bold(X) dot bold(W) + bold(B) &= mat(
    bold(x)_1, bold(x)_2, dots, bold(x)_batch;
  )^top dot bold(W) + bold(B)\
  &= mat(
    bold(x)_1 dot bold(W), bold(x)_2 dot bold(W), dots, bold(x)_batch dot bold(W);
  )^top + bold(B)\
  &= mat(
    bold(x)_1 dot bold(W) + bold(b),thick bold(x)_2 dot bold(W) + bold(b),thick dots,thick bold(x)_batch dot bold(W) + bold(b);
  )^top\
  &= mat(
    bold(y)_1, bold(y)_2, dots, bold(y)_batch;
  )^top\
  &= bold(Y)
$

== 行列表現での活性化関数の扱い
活性化関数はAffine層とは別と考えることができる。実際の深層学習フレームワークでは一緒に利用できるように実装されているが、ここでは別々に考えることにする。ReLUやシグモイドなどの関数は、行列の個々の要素に対して適用すれば良い。
$
  [bold(Y)]_(i j) = f_text("ReLU")([bold(X)]_(i j))\
  [bold(Y)]_(i j) = f_text("sigmoid")([bold(X)]_(i j))\
$

Softmax関数などのデータ全体を用いる関数は、バッチ内のデータごとに適用すれば良い。行列表現では行ごとということになる。\
$
  [bold(Y)]_(i j) = frac(exp([bold(X)]_(i j)), sum_k exp([bold(X)]_(i k)))
$

#pagebreak()

= ニューラルネットワーク
多層パーセプトロンを実装するだけでは、意図した出力を得ることはできない。適正なパラメータを与えなければならない。しかし、人間が正確なパラメータを与えることはおおよその場合、不可能である。このパラメータを自動的に調整する方法が必要である。深層学習とはつまるところ、パラメータを自動的に調整することである。\

多層パーセプトロンに与えるデータを$bold(x)^((text("train")))$、それを与えた場合に出力されるべきデータを$bold(y)^((text("train")))$、実際に出力されたデータを$bold(y)^((text("pred")))$とする。$bold(y)^((text("train")))$と$bold(y)^((text("pred")))$は大きさ$C$のベクトルとする。\

== 損失関数
損失関数$cal(L)$は、実際に出力されるデータ$bold(y)^((text("train")))$と与えられたデータ$bold(y)^((text("pred")))$の差を表す関数である。これは、パラメータを調整するための指標となる。多層パーセプトロンは数学的には$bold(x)^((text("train")))$だけでなく、用いるパラメータすべてを引数にとる多変数関数とみなせる。なので、その出力を引数に持つ、損失関数もそれらを引数に持つ多変数関数とみなせる。ここで、$bold(y)^((text("train")))$は決められた定数とみなす。
$

  cal(L) :& RR^(abs(bold(x)^((text("train"))))) times RR^(text("parameters")) -> RR\
$

いくつかの損失関数があるが、ここでは代表的なものを紹介する。\

=== 平均二乗誤差
平均二乗誤差は、$bold(y)_text("pred")$と$bold(y)_text("train")$の個々の差を二乗して足し合せて平均をとったものである。これは、出力の値が連続的な場合に用いられる。\
$
  cal(L)_text("MSE") = 1/C abs(abs(bold(y)^((text("pred"))) - bold(y)^((text("train")))))^2
$

=== クロスエントロピー誤差
クロスエントロピー誤差は、$bold(y)_text("pred")$と$bold(y)_text("train")$を確率分布とみなして、その確率分布の誤差を表すものである。分類問題において使われることが多い。Softmax関数と組み合わせて用いることが多い。\

$
  cal(L)_text("CrossEntropy") = -sum_(1<=j<=C) y_i^((text("train"))) log(y_i^((text("pred"))))
$

#colbreak()

== 確率的勾配降下法（SGD）
損失を最小化することが、深層学習の目標である。あるパラメータ$p$を更新することを考えよう。損失関数$cal(L)$は先述した通り,$p$を引数に持つ。$p$についての$cal(L)$のグラフを書くと以下のようになる。
#let f(x) = (1 / x) * calc.sin((x - 10)) + calc.pow((x - 10),4) - 8 * calc.pow((x - 10),2) + 4 * (x - 10) + 40

#align(center)[
#cetz.canvas({
  plot.plot(
    size: (8, 3),
    axis-style: "school-book",
    x-min: 4,
    x-label: [$p$],
    y-label: [$cal(L)$],
    x-max: 15,
    y-min: -3,
    y-max: 100,
    x-tick-step: 6.5,
    y-tick-step: none,
    x-format: (v) => if v < 7 {
      return $p_1$
    } else {
      return $p_2$
    },
    {
      plot.add(domain: (1, 18),samples: 100, f, style: (stroke: 2pt + blue))
      plot.add(((13,0),(13,f(13))), mark: none, style: (stroke:(
        dash: "dotted",
      )))
      plot.add(((6.5,0),(6.5,f(6.5))), mark: none, style: (stroke:(
        dash: "dotted",
      )))
    }
  )
})
]

実際は$p$に対する$cal(L)$の形はわからない。しかし、$p$での傾きを知ることはできる。これは後述する。$p_1$のように傾きが負のときは、$p$を増やすことで損失が減る。$p_2$のように傾きが正のときは、$p$を減らすことで損失が減る。したがって、$p$を更新する際には、傾きに応じて$p$を増減させれば良い。これを勾配降下法という。つまり、$p$をの更新は以下のように行う。
$
  p arrow.l p - eta (partial cal(L))/(partial p)\
$

ここで、$eta$は学習率とよばれる定数である。これは、$p$の更新幅を決めるものである。$eta$が大きすぎると、最適なパラメータを通り過ぎてしまうことがある。逆に小さすぎると、最適なパラメータにたどり着くまでに時間がかかる。この更新方法を確率的勾配降下法（SGD）とよぶ。\

現在、パラメータの更新にはSGDに様々な工夫を加えたものが用いられている。しかし、偏微分を考えるという部分は変わらない。\

== 誤差逆伝播法
さて、SGDを用いるためには、$cal(L)$の偏微分を計算する必要がある。$cal(L)$は多層パーセプトロンの出力を引数に持つ多変数関数である。したがって、$cal(L)$の偏微分は連鎖律を用いて計算できる。

=== 数値微分
連鎖律を用いる以外には、数値微分を用いる方法がある。本書では、連鎖律を用いることを前提としているので、詳しくは述べないが紹介だけしておく。数値微分は、$cal(L)$の入力を少しだけ変えて、出力の変化をみることで偏微分を求める方法である。$p$に関しての偏微分は以下のように表される。そもそも、偏微分の定義は以下のようなものであった。
$
  (partial cal(L))/(partial p) = lim_(delta -> 0) (cal(L)(p + delta) - cal(L)(p))/ delta
$
これを簡単に求められない、つまり四則演算を用いて計算することができないのは$delta$を$0$に近づけるためである。そこで、$delta$を極めて小さな値にし、
$
  (cal(L)(p + delta) - cal(L)(p))/ delta
$
を計算することで、近似的に偏微分を求めることができる。このように、$delta$を用いて偏微分を近似的に求める方法を数値微分とよぶ。上の式は、$delta$が正なら前方差分、$delta$が負なら後方差分とよばれる。しかし、求めたいのは$p$における偏微分であるのでこれらの計算は誤差が大きい。したがって、以下のように$p$を中心にした差分を計算することが多い。これを中心差分とよぶ。中心差分は前方差分や後方差分よりも精度がいいことは解析的に示すことができる。
$
  frac(cal(L)(p + delta) - cal(L)(p - delta), 2 delta)
$
すべてのパラメータに対して、$delta$を用いて偏微分を近似的に求めることができればいいが、パラメータの数は膨大であるため、計算量が膨大になってしまう。また、より精度のいい5点公式や7点公式などもある。

=== 実数の逆伝播
行列で逆伝播を考える前に、実数の逆伝播法を考える。例えばある実数を返す関数$cal(L)$#footnote[表現からも分かる通り、損失関数を念頭に置いている。]に関してある出力$y$による偏微分係数が既知であるとする。そして入力$x$に関して、
$
  y = w x + b
$
が成立するとする。$w,b$は更新対象のパラメータである。勾配降下法を用いるためにはこれらのパラメータに関する偏微分係数が必要である。そのためにまず、$y$の偏微分を求める。
$
  (partial y)/(partial x) = w wide
  (partial y)/(partial w) = x wide
  (partial y)/(partial b) = 1
$
次にこれらの関係をまとめると、
$
  x quad arrow.bar quad y quad arrow.bar quad cal(L)(y)\
  w quad arrow.bar quad y quad arrow.bar quad cal(L)(y)\
  b quad arrow.bar quad y quad arrow.bar quad cal(L)(y)\
$
であるので、連鎖律を用いて以下のように表すことができる。
$
  (partial cal(L))/(partial x) &= (partial cal(L))/(partial y) (partial y)/(partial x) &=& w (partial cal(L))/(partial y)\
  (partial cal(L))/(partial w) &= (partial cal(L))/(partial y) (partial y)/(partial w) &=& x (partial cal(L))/(partial y)\
  (partial cal(L))/(partial b) &= (partial cal(L))/(partial y) (partial y)/(partial b) &=& (partial cal(L))/(partial y)\
$
ここで、$display((partial cal(L))/(partial y))$は既知であるので,入力とパラメータに関する偏微分係数を求めることができた。$display((partial cal(L))/(partial x))$はなぜ求めたのだろうか。これは$x$が前の層の出力であるからである。つまり、$display((partial cal(L))/(partial x))$はこの前の層にとっては既知とされた$display((partial cal(L))/(partial y))$に対応するのだ。\

=== Affine層の逆伝播
ある行列$bold(X) in MM^(n times m)$について$display(partial/(partial bold(X)))$を次のように定義する。
$
  display(partial/(partial bold(X))) colon= mat(
    display(partial/(partial x_11)), display(partial/(partial x_12)), dots, display(partial/(partial x_(1m)));
    display(partial/(partial x_21)), display(partial/(partial x_22)), dots, display(partial/(partial x_(2m)));
    dots.v, dots.v, dots.down, dots.v;
    display(partial/(partial x_(n 1))), display(partial/(partial x_(n 2))), dots, display(partial/(partial x_(n m)));
  )\
$

$display(partial/(partial bold(X)))$自身も形式上は$n times m$の行列であり、ドット積は形式的に行える。左側から行われた場合は、例えば$display(partial/(partial x_11)) y$のように書くことができるが、$y$を$x_11$で偏微分するという意味になる。\

これを踏まえて、バッチ付きAffine層の逆伝播を考える。Affine層の出力$bold(Y)$についての実数関数$cal(L)$の偏微分$display((partial cal(L))/(partial bold(Y)))$は既知であるとする。なお、要素を書き下すと以下のようになることに注意されたい。
$
  (partial cal(L))/(partial bold(Y)) colon= mat(
    (partial cal(L))/(partial y_11), (partial cal(L))/(partial y_12), dots, (partial cal(L))/(partial y_(1 m));
    (partial cal(L))/(partial y_21), (partial cal(L))/(partial y_22), dots, (partial cal(L))/(partial y_(2m));
    dots.v, dots.v, dots.down, dots.v;
    (partial cal(L))/(partial y_(n 1)), (partial cal(L))/(partial y_(n 2)), dots, (partial cal(L))/(partial y_(n m));
  )
$
Affine層は以下のような演算を行うことは既に述べた通りである。
$
  bold(Y) = bold(X) dot bold(W) + bold(B)
$
$y_(i j)$について以下のように書き下せる。
$
  y_(i j) = sum_(1<=alpha<=n) x_(i alpha) w_(alpha j) + b_j
$
つぎに、両辺を$w_(p q)$と$x_(p q)$で偏微分すると、
$
  (partial y_(i j))/(partial w_(p q)) &= cases(
    thick x_(i p) & quad text("if") quad j = q,
    thick 0 & quad text("otherwise"),
  )\
  (partial y_(i j))/(partial x_(p q)) &= cases(
    thick w_(q j) & quad text("if") quad i = p,
    thick 0 & quad text("otherwise"),
  )
$
となる.$X$の行と$W$の列は固定されているので、式にそれらが出現しないと偏微分係数は$0$になる。そして、$cal(L)$がどのように$w_(p q)$に依存しているかを示すと、
$
  w_(p q) quad arrow.bar quad y_(i j) thick (1 <= forall i <= n, 1<=forall j<=m) quad arrow.bar quad cal(L)\
$
となっているので、
$
  [(partial cal(L))/(partial bold(W))]_(p q) = (partial cal(L))/(partial w_(p q)) &= sum_(1<=i<=n)sum_(1<=j<=m)(partial cal(L))/(partial y_(i j)) (partial y_(i j))/(partial w_(p q))\
  &= sum_(1<=i<=n) (partial cal(L))/(partial y_(i q)) x_(i p)\
  &= sum_(1<=i<=n) [(partial cal(L))/(partial bold(Y))]_(i q) [bold(X)]_(i p)\
  &= sum_(1<=i<=n) [bold(X)^top]_(p i) [(partial cal(L))/(partial bold(Y))]_(i q)\
$
が成立する。これはまさに、行列のドット積の定義であるので
$
  (partial cal(L))/(partial bold(W)) = bold(X)^top dot (partial cal(L))/(partial bold(Y))
$
が成立する。同様に
$
  (partial cal(L))/(partial bold(X)) = (partial cal(L))/(partial bold(Y)) dot bold(W)^top
$
も成立する。$bold(X), bold(W), display((partial cal(L))/(partial bold(Y)))$は既知であるので、重みの偏微分係数を求めることができた。
さて、バイアスについては、以下のような依存関係がある。
$
  b_j quad arrow.bar quad y_(i j) thick (1 <= forall i <= n) quad arrow.bar quad cal(L)\
$
したがって、$b_j$について偏微分すると
$
  (partial cal(L))/(partial b_j) &= sum_(1<=i<=n) (partial cal(L))/(partial y_(i j)) (partial y_(i j))/(partial b_j)\
  &= sum_(1<=i<=n) (partial cal(L))/(partial y_(i j))\
$
となる。$b_j$は$y_(i j)$に依存しているので、$j$番目の列の和をとることになる。したがって、逆伝播は以下のように表せる。
$
  (partial cal(L))/(partial bold(b)) = sum_(1<=i<=n) [(partial cal(L))/(partial bold(Y))]_(i)\
$
ここで、$[(partial cal(L))/(partial bold(Y))]_(i)$は$(partial cal(L))/(partial bold(Y))$の$i$行目を表す。

=== ReLU関数の逆伝播
ReLU関数は非常に簡単に逆伝播できる。ReLU関数は以下のように定義されていた。
$
  y_(i j) = cases(
    thick x_(i j) & quad text("if") quad x_(i j) >= 0,
    thick 0 & quad text("otherwise"),
  )
$
これを$x_(p q)$について偏微分すれば
$
  (partial y_(i j))/(partial x_(p q)) = cases(
    thick 1 & quad text("if") quad i = p and j = q and x_(i j) >= 0,
    thick 0 & quad text("otherwise"),
  )
$
となる。よって、
$
  (partial cal(L))/(partial x_(p q)) &= sum_(1<=i<=n) sum_(1<=j<=m) (partial cal(L))/(partial y_(i j)) (partial y_(i j))/(partial x_(p q))\
  &= cases(
    thick 1 & quad text("if") x_(p q) >= 0,
    thick 0 & quad text("otherwise"),
  )
$
となる。ReLU関数自体が、行列の各要素ごとに適応されるため、その逆伝播も行列の各要素ごとに適応される。したがって、ReLU関数の逆伝播は行列の計算を伴わず非常に簡単である。\

=== Softmax関数の逆伝播
Softmax関数は、行列の各行ごとに適応される。したがって、Softmax関数の逆伝播も行列の各行ごとに考える。この節に限り、入力を$bold(x)$、出力を$bold(y)$とする。ともに大きさ$C$である。すると、Softmax関数は以下のように定義されていた。
$
  y_(j) = frac(exp(x_(j)), sum_k exp(x_(k)))
$
右辺の分母を$S$とすると、$y_(j)$を$x_q$で偏微分すると
$
  (partial y_(j))/(partial x_q) &= cases(
    thick display((exp(x_j) S - exp(x_j){0+exp(x_j)})/ S^2) & quad text("if") quad j = q,
    thick display(- (exp(x_j){0+exp(x_j)}) / S^2) & quad text("otherwise"),
  )\
  &= cases(
    thick display(exp(x_j)/S - exp(x_j)^2/S^2) & quad text("if") quad j = q,
    thick display(-exp(x_j)^2/S^2) & quad text("otherwise"),
  )\
  &= cases(
    thick y_(j) - y_(j)^2 & quad text("if") quad j = q,
    thick -y_(j) y_(q) & quad text("otherwise"),
  ) wide = y_j (delta_(j q) - y_q)\
$
となる。ここで、$delta_(j q)$はKroneckerのデルタである。$j=q$のときは$1$、それ以外は$0$である。依存関係は以下のように表せる。
$
  x_q quad arrow.bar quad y_j thick (1 <= forall j <= C) quad arrow.bar quad cal(L)
$
連鎖律を用いれば以下のように求められる。
$
  (partial cal(L))/(partial x_q) &= sum_(1<=j<=C) (partial cal(L))/(partial y_j) (partial y_j)/(partial x_q)\
  &= sum_(1<=j<=C\ j eq.not q) (partial cal(L))/(partial y_j) (-y_j y_q) + (partial cal(L))/(partial y_q) (y_q - y_q^2)\
  &= sum_(1<=j<=C) (partial cal(L))/(partial y_j) (-y_j y_q) + (partial cal(L))/(partial y_q) y_q\
  &= sum_(1<=j<=C) [(partial cal(L))/(partial bold(y))]_j [
    mat(
      y_1 , , , ;
      , y_2 , , ;
      , , dots.down, ;
      , , , y_C;
    )]_(j q) - sum_(1<=j<=C) [(partial cal(L))/(partial bold(y))]_j [
    mat(
      y_1 y_1, y_1 y_2, dots, y_1 y_C;
      y_2 y_1, y_2 y_2, dots, y_2 y_C;
      dots.v, dots.v, dots.down, dots.v;
      y_C y_1, y_C y_2, dots, y_C y_C;
    )]_(j q)\
$
やや強引な式変形を行うと、簡単に求めることができる行列を用いることができる。以上から、
$
  (partial cal(L))/(partial bold(x)) &= (partial cal(L))/(partial bold(y)) dot { mat(
    y_1 , , , ;
    , y_2 , , ;
    , , dots.down, ;
    , , , y_C;
  ) - mat(
    y_1 y_1, y_1 y_2, dots, y_1 y_C;
    y_2 y_1, y_2 y_2, dots, y_2 y_C;
    dots.v, dots.v, dots.down, dots.v;
    y_C y_1, y_C y_2, dots, y_C y_C;
  )}\
  &= (partial cal(L))/(partial bold(y)) dot { text("diag"(bold(y))) - bold(y)^top dot bold(y) }\
$
ここで、$text("diag"(bold(y)))$は$bold(y)$の対角行列を表す。numpyなどのライブラリにはdiagが実装されていることが多い。

=== クロスエントロピー誤差の逆伝播
レイヤと同じように損失関数にも逆伝播が必要である。逆伝播の開始位置ともいえるだろう。クロスエントロピー誤差は以下のように定義されていた。ただし、モデルの出力を$bold(x)$、教師データを$bold(t)$とする。
$
  cal(L) = -1/C sum_(1<=j<=C) t_j log(x_j))
$
Softmax関数と同様に行ごとに適応される。逆伝播はすぐに求めることができる。
$
  (partial cal(L))/(partial y_j) = - 1/C t_j/x_j\
$

=== Softmax関数とクロスエントロピー誤差の組み合わせ
Softmax関数とクロスエントロピー誤差は、組み合わせて用いることが多い。というのも、Softmax関数は確率分布を表し、クロスエントロピー誤差は確率分布の誤差を表すからである。Softmax関数とクロスエントロピー誤差を組み合わせて逆伝播を求めてみる。ただし、Softmax関数への入力を$bold(x)$、出力を$bold(y)$クロスエントロピー誤差の教師データを$bold(t)$とする。\
$
  (partial cal(L))/(partial x_q) &= sum_(1<=j<=C) (partial cal(L))/(partial y_j) (partial y_j)/(partial x_q)\
  &= sum_(1<=j<=C) (partial cal(L))/(partial y_j) (-y_j y_q) + (partial cal(L))/(partial y_q) y_q\
  &= sum_(1<=j<=C) (-1/C t_j/y_j) (-y_j y_q) + (-1/C t_q/y_q) y_q\
  &= y_q / C (sum_(1<=j<=C) t_j - t_q) \
  &= (y_q - t_q) / C
$
ここで、$t_j$はクロスエントロピー誤差の教師データである。$t_j$は正規化されているので、$sum_(1<=j<=C) t_j = 1$である。このように、Softmax関数とクロスエントロピー誤差を組み合わせることで、非常に簡単に逆伝播を求めることができる。加えて、Softmax関数もクロスエントロピー誤差も行列計算不可能にも関わらず、この逆伝播はバッチ付きの行列計算で表現できる。
$
  (partial cal(L))/(partial bold(X)) &= 1/C (bold(Y) - bold(T))
$

= ニューラルネットワークの工夫
前章までで、ニューラルネットワークの基本的な部分を学んだ。ここからは、ニューラルネットワークをより効率的に学習させるための工夫を紹介する。

== Optimizer
Optimizerは、パラメータを更新するためのアルゴリズムである。SGDはOptimizerの一つである。ここでは、SGDにさまざまな工夫を加えたOptimizerを紹介する。この章ではあるパラメータ$bold(PP)$を更新することを考える。

=== Momentum
Momentumは、SGDの更新に過去の更新を引き継いだ慣性のようなものを加えたものである。前回までの更新を$bold(M)$に保存し、新たな更新では勾配と$bold(M)$の両方を用いる。
$
  bold(M) & arrow.l alpha bold(M) - eta (partial cal(L))/(partial bold(P))\
  bold(P) & arrow.l bold(P) + bold(M)\

$

=== RMSProp
RMSPropは、SGDの更新に対して学習率を調整する機能を付け加えたものである。更新分が一定だと、最小に近づくにつれ、更新幅が必要な幅より大きくなってしまう問題を解決するため、更新幅の分だけ学習率を割り引く。更新幅は指数移動平均を用いて計算し、ある程度過去の更新幅を引き継ぐ。
$
  bold(G) & arrow.l beta bold(G) + (1 - beta) (partial cal(L))/(partial bold(P)) circle.small (partial cal(L))/(partial bold(P))\
  bold(P) & arrow.l bold(P) - eta/sqrt(bold(G)) (partial cal(L))/(partial bold(P)) \
$

パラメータごとに学習率を調整するために、アダマール積とアダマール除算を用いている。

=== Adam
Adamは、MomentumとRMSPropを組み合わせたものである。
$
  bold(M) & arrow.l beta_1 bold(M) + (1 - beta_1) (partial cal(L))/(partial bold(P))\
  bold(G) & arrow.l beta_2 bold(G) + (1 - beta_2) (partial cal(L))/(partial bold(P)) circle.small (partial cal(L))/(partial bold(P))\
  bold(P) & arrow.l bold(P) - eta/sqrt(bold(G)) bold(M)
$

== レイヤ
Affine層や活性化関数層以外の、レイヤを追加することで学習を加速あるいは、精度を上げることができる。ここでは、代表的なレイヤを紹介する。\

=== Batch Normalization
==== 順伝播
学習時はバッチを用いて複数のデータを同時に学習する。この際、バッチ内のデータに偏りが存在する場合があり、学習が進まないことがある。また、途中の層の出力が極端になってしまうことがある。これを防ぐために、バッチ内のデータについて、各特徴量を正則化する。その後、学習可能なパラメータを用いて、正則化したデータを変換する。これをBatch Normalization（以下、BN）とよぶ。入力が$bold(X) in MM^(B times C)$の行列である場合、正則化を行うのは列ごとであることに注意されたい。

ここでは$bold(X)$の$i$行目ベクトルを$bold(x)^((i))$とする。ここでは、要素$x^((i))_j$がどのように正則化されるかを考える。その際、$j$列目のことのみを考えればよい。$j$行目の平均を$mu_j$、分散を$sigma_j$とし、中間出力を$bold(z)^((i))$、最終出力を$bold(y)^((i))$とする。すると、BNは以下のように行われる。
$
  mu_j &= 1/B sum_(1<=k<=B) x^((k))_j\
  sigma_j &= 1/B sum_(1<=k<=B) (x^((k))_j - mu_j)^2\
  z^((i))_j &= (x^((i))_j - mu_j) / sigma_j\
  y^((i))_j &= gamma_j z^((i))_j + beta_j\
$

ここで、$gamma_j$と$beta_j$は学習可能なパラメータで、行ごとに別の値が適応される。正則化後のデータを適切にスケールするためのものである。$mu_j$、$sigma_j$は推論時に列ごとに計算される。つまり、実際には$bold(gamma), bold(beta), bold(mu), bold(sigma)$はベクトルである。

==== 逆伝播
次に、学習可能なパラメータである$bold(gamma)$と$bold(beta)$と、$bold(X)$の逆伝播を考える。ここでも、$(partial cal(L))/(partial bold(Y))$は既知とする。まずは、$bold(Y)$から$bold(Z)$までの逆伝播を考える。まず、$gamma_j, beta_j, z^((i))_j$の損失関数$cal(L)$に対する依存関係を整理しよう。
$
  gamma_j quad arrow.bar quad y^((k))_j thick (1 <= forall k <= B) quad arrow.bar quad cal(L)\
  beta_j quad arrow.bar quad y^((k))_j thick (1 <= forall k <= B) quad arrow.bar quad cal(L)\
  z^((i))_j quad arrow.bar quad y^((i))_j quad arrow.bar quad cal(L)\
$
つまり、
$
  (partial cal(L))/(partial z^((i))_j) &= (partial cal(L))/(partial y^((i))_j) (partial y^((i))_j)/(partial z^((i))_j) =  (partial cal(L))/(partial y^((i))_j) gamma_j\
  (partial cal(L))/(partial gamma_j) &= sum_(1<=k<=B) (partial cal(L))/(partial y^((k))_j) (partial y^((k))_j)/(partial gamma_j)\
  &= sum_(1<=k<=B) (partial cal(L))/(partial y^((k))_j) z^((k))_j\
  (partial cal(L))/(partial beta_j) &= sum_(1<=k<=B) (partial cal(L))/(partial y^((k))_j) (partial y^((k))_j)/(partial beta_j)\
  &= sum_(1<=k<=B) (partial cal(L))/(partial y^((k))_j)\
$
である。つまり、ベクトル表現を用いると
$
  (partial cal(L))/(partial bold(gamma)) &= sum_(1<=k<=B) (partial cal(L))/(partial bold(y)^((k))) circle.small bold(z)^((k))\
  (partial cal(L))/(partial bold(beta)) &= sum_(1<=k<=B) (partial cal(L))/(partial bold(y)^((k)))\
$
となる。次に、$bold(Z)$から$bold(X)$までの逆伝播を考える。注意すべき点は$x^((i))_j$が$mu$や$sigma$に寄与しており、それがすべての$z^((k))_j (1 <= forall k <= B)$に寄与していることである。計算しやすさのため、$u^((i))_j colon=x^((i))_j - mu_j$を定義する。すると、$sigma_j$は
$
  sigma_j &= 1/B sum_(1<=k<=B) (u^((k))_j)^2\
$
と表される。$u^((i))_j$を用いて関係をまとめると以下のようになる。ただしこの図で$k$は$1<=k<=B and k eq.not i$を満たすすべての$k$である。

#align(center)[
  #set text(size: 12pt, weight: 500)
  #diagram(
  node-stroke: none,
  edge-corner-radius: none,
  node-inset: 5pt,
  spacing: (20pt, 25pt),
  node-shape: "circle",

  // Nodes
  node((-1,0), [$x^((i))_j$], name: <x>, width: 1.0cm),
  node((0,1), [$mu_j$], name: <mu>, width: 1.0cm),
  node((1,0), [$u^((i))_j$], name: <ui>, width: 1.0cm),
  node((1,1), [$u^((k))_j$], name: <uk>, width: 1.0cm),
  node((2,0.5), [$sigma^2_j$], name: <sig2>, width: 1.0cm),
  node((3,0.5), [$sigma_j$], name: <sig>, width: 1.0cm),
  node((4,0), [$z^((i))_j$], name: <zi>, width: 1.0cm),
  node((4,1), [$z^((k))_j$], name: <zk>, width: 1.0cm),
  node((5,0.5), [$cal(L)$], name: <l>, width: 1.0cm),

  edge(<x>, <mu>, "|->"),
  edge(<x>, <ui>, "|->"),
  edge(<mu>, <ui>, "|->"),
  edge(<mu>, <uk>, "|->"),
  edge(<ui>, <sig2>, "|->"),
  edge(<uk>, <sig2>, "|->"),
  edge(<ui>, <zi>, "|->"),
  edge(<uk>, <zk>, "|->"),
  edge(<sig2>, <sig>, "|->"),
  edge(<sig>, <zi>, "|->"),
  edge(<sig>, <zk>, "|->"),
  edge(<zi>, <l>, "|->"),
  edge(<zk>, <l>, "|->"),
  )
]
$u^((i))_j$に対しては、$x^((i))_j$が$mu_j$を介してだけではなく直接作用するので、$u^((i))_j$と$z^((i))_j$を他の$u^((k))_j$や$z^((k))_j$と区別する必要がある。逆に、$u^((k))_j$までは区別する必要はないのでこれからは$k$は$i$も含めた任意の添字として扱う。まずは、$sigma_j$について考える。
$
  z^((k))_j = (u^((k))_j) / sigma_j quad
  &=> quad (partial z^((k))_j)/(partial sigma_j) = - (u^((k))_j) / sigma_j^2\
  (partial sigma_j)/(partial sigma^2_j) &= 1/(2 sigma_j)\
  therefore (partial cal(L))/(partial sigma^2_j) &= sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j) (partial z^((k))_j)/(partial sigma_j) (partial sigma_j)/(partial sigma^2_j)\
  &= - 1/(2 sigma^3_j) sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j) u^((k))_j\
$
次に、$u^((k))_j$について考えると、
$
  sigma_j = 1/B sum_(1<=k<=B) (u^((k))_j)^2 quad => quad (partial sigma^2_j)/(partial u^((k))_j) &= 2/B u^((k))_j\
$
である。また、$z^((k))_j = u^((k))_j / sigma_j$から、$u^((i))_j$の$sigma_j$に対する依存を一旦無視すると
$
  lr((partial z^((k))_j)/(partial u^((k))_j)|)_text("direct")  &= 1/sigma_j\
$
であるので、
$
  (partial z^((k))_j)/(partial u^((k))_j) &= lr((partial z^((k))_j)/(partial u^((k))_j)|)_text("direct") + (partial z^((k))_j)/(partial sigma^2_j) (partial sigma^2_j)/(partial u^((k))_j)\
  &= 1/sigma_j + 2/B u^((k))_j (partial z^((k))_j)/(partial sigma^2_j)\
  therefore (partial cal(L))/(partial u^((k))_j) &= (partial cal(L))/(partial z^((k))_j) 1/sigma_j + 2/B u^((k))_j (partial cal(L))/(partial sigma^2_j)
$
となる。最後に$u^((k))_j$に対する$x^((i))_j$への逆伝播を考える。しかし、$mu_j$がすべての$u^((k))_j$に寄与していることを考慮する必要がある。
$
  u^((k))_j = x^((k))_j - mu_j quad &=> quad (partial u^((k))_j)/(partial mu_j) = -1\
  therefore (partial cal(L))/(partial mu_j) &= sum_(1<=k<=B) (partial cal(L))/(partial u^((k))_j) (partial u^((k))_j)/(partial mu_j)\
  &= - sum_(1<=k<=B) (partial cal(L))/(partial u^((k))_j)\
  &= - sum_(1<=k<=B) [(partial cal(L))/(partial z^((k))_j) 1/sigma_j + 2/B u^((k))_j (partial cal(L))/(partial sigma^2_j)]\
  &= - 1/sigma_j sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j) - 2/B (partial cal(L))/(partial sigma^2_j) sum_(1<=k<=B) u^((k))_j \
$
ここで、$sum_(1<=k<=B) u^((k))_j$について考えてみる。
$
  sum_(1<=k<=B) u^((k))_j &= sum_(1<=k<=B) (x^((k))_j - mu_j)\
  &= sum_(1<=k<=B) x^((k))_j - B mu_j\
  &= sum_(1<=k<=B) x^((k))_j - sum_(1<=k<=B) x^((k))_j\
  &= 0\
$
したがって、
$
  (partial cal(L))/(partial mu_j) &= - 1/sigma_j sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j)\
$
そして、$x^((i))_j$は直接$u^((i))_j$に寄与している他、$mu_j$を通して$u^((k))_j$に寄与している。つまり、
$
  mu_j = 1/B sum_(1<=k<=B) x^((k))_j quad &=> quad (partial mu_j)/(partial x^((i))_j) = 1/B\
  therefore (partial cal(L))/(partial x^((i))_j) &= (partial cal(L))/(partial u^((i))_j) lr((partial u^((i))_j)/(partial x^((i))_j)|)_text("direct") + (partial cal(L))/(partial mu_j) (partial mu_j)/(partial x^((i))_j)\
  &= (partial cal(L))/(partial u^((i))_j) + 1/B (partial cal(L))/(partial mu_j)\
  &= (partial cal(L))/(partial u^((i))_j) - 1/(B sigma_j) sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j)\
$
となる。第1項について詳しく考える。$z^((k))_j = u^((k))_j / sigma_j$を念頭に考えると、
$
  (partial cal(L))/(partial u^((i))_j) &= (partial cal(L))/(partial z^((i))_j) 1/sigma_j + 2/B u^((i))_j (partial cal(L))/(partial sigma^2_j)\
  &= (partial cal(L))/(partial z^((i))_j) 1/sigma_j + 2/B u^((i))_j (-1/(2 sigma^3_j) sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j) u^((k))_j)\
  &= (partial cal(L))/(partial z^((i))_j) 1/sigma_j - 1/sigma_j 1/B u^((i))_j/sigma_j sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j) u^((k))_j/sigma_j\
  &= (partial cal(L))/(partial z^((i))_j) 1/sigma_j - 1/sigma_j z^((i))_j/B sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j) z^((k))_j\
$
つまり、以上をまとめると
$
  (partial cal(L))/(partial x^((i))_j) &= (partial cal(L))/(partial z^((i))_j) 1/sigma_j - 1/sigma_j z^((i))_j/B sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j) z^((k))_j\
  &= (partial cal(L))/(partial z^((i))_j) 1/sigma_j - 1/sigma_j z^((i))_j/B sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j) z^((k))_j - 1/(B sigma_j) sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j)\
  &= 1/sigma_j [
    (partial cal(L))/(partial z^((i))_j) - z^((i))_j/B sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j) z^((k))_j - 1/B sum_(1<=k<=B) (partial cal(L))/(partial z^((k))_j)
  ]\
  &= gamma_j/sigma_j [
    (partial cal(L))/(partial y^((i))_j) - z^((i))_j/B sum_(1<=k<=B) (partial cal(L))/(partial y^((k))_j) z^((k))_j - 1/B sum_(1<=k<=B) (partial cal(L))/(partial y^((k))_j)
  ]\
$
となる。最後の等号は、$(partial cal(L))/(partial z^((i))_j) = gamma_j (partial cal(L))/(partial y^((i))_j)$を用いている。さて、$gamma_j$や$beta_j$の勾配を思い出すと、
$
  (partial cal(L))/(partial x^((i))_j) &= gamma_j/sigma_j [
    (partial cal(L))/(partial y^((i))_j) - z^((i))_j/B (partial cal(L))/(partial gamma_j) - 1/B (partial cal(L))/(partial beta_j)
  ]\
$
となる。この式はすべての$j$列に対して成立するのでベクトル表現を用いれば以下のように整理できる。
$
  (partial cal(L))/(partial bold(x)^((i))) = bold(gamma)/bold(sigma) circle.small [
    (partial cal(L))/(partial bold(y)^((i))) - 1/B { bold(z)^((i)) circle.small (partial cal(L))/(partial bold(gamma)) + (partial cal(L))/(partial bold(beta))}
  ]\
$

さらに成分がすべて$1$で大きさ$B$の列ベクトル$bold(1)_B$を用いて行列表現に拡張すると以下のようになる。
$
  (partial cal(L))/(partial bold(X)) = (bold(1)_B dot bold(gamma)/bold(sigma)) circle.small [
    (partial cal(L))/(partial bold(Y)) - 1/B { bold(Z) circle.small (bold(1)_B dot (partial cal(L))/(partial bold(gamma))) + bold(1)_B dot (partial cal(L))/(partial bold(beta))}
  ]\
$