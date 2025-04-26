#let parameters_splited(parameters) = parameters.map(it => {
  let parts = it.split(":");
  ([
    #strong([#parts.at(0)])
    ], [#parts.at(1)])
}).flatten()


#let init(header-left: "", header-right: "", footer-right: "", body) = {
  show math.equation: it => h(0.25em, weak: true) + it + h(0.25em, weak: true)

  set text(
    font: ("Hiragino Kaku Gothic ProN"),
    lang: "ja",
    size: 11pt,
    weight: 300,
    top-edge: 0.88em,
  )

  set page(
    paper: "a4",
    margin: (top: 1.5cm, bottom: 1.5cm, x: 1.2cm),
    numbering: "1",
    number-align: right,
    header: [
      #set text(size: 9pt);
      #header-left
      #h(1fr)
      #header-right
    ],
  )

  set heading(numbering: "1.1.")
  body
}


#let topbar(
  titlefield: [
    科目名
    #v(-9mm)
    #set text(size: 32pt, weight: 500);
    #h(-4pt);
    レポートテンプレート
    #set text(size: 12pt, weight: 500);
    サブタイトル
  ],
  parameters: (
    "名前:俺",
    "所属:情報工学系",
    "学籍番号:？？？？",
  ),
  body,
) = {
  v(0.3cm)
  grid(
    columns: (2.5fr, 1fr),
    gutter: 1cm,
  )[#titlefield][
    #v(1cm)
    #table(
      columns: (auto, 1fr),
      align: (left, right),
      stroke: none,
      ..parameters_splited(parameters)
    )
  ]
  v(0.5cm)
  body
}

#let default_format(n, m) = [
  #set text(size: 12pt, weight: 500)
  （#h(1pt)#m#h(1pt)）
]

#let toi(
  format: (n) => [
    #set text(size: 18pt, weight: 500);
    第#n 問
  ],
) = {
  let toi_counter = counter("num_toi")
  let subtoi_counter = counter("num_subtoi")
  toi_counter.step()
  context {
    let should_reset = state("toi_reset", true).get()

    if should_reset {
      subtoi_counter.update(0)
    }
    let n = toi_counter.get().at(0)
    format(n)
  }
}

#let subtoi(
  format: default_format
) = {
  let toi_counter = counter("num_toi")
  let subtoi_counter = counter("num_subtoi")
  subtoi_counter.step()
  context {
    let n = toi_counter.get().at(0)
    let m = subtoi_counter.get().at(0)
    format(n, m)
  }
}

#let thesubtoi(
  format: (n, m) => default_format(n, m)
) = {
  let toi_counter = counter("num_toi")
  let subtoi_counter = counter("num_subtoi")
  context {
    let n = toi_counter.get().at(0)
    let m = subtoi_counter.get().at(0)
    format(n, m)
  }
}

#let answerunderline(
  baseline: 1pt,
  padding: 0.5em,
  offset: 1em,
  subsize: 0.8em,
  body,
  sub
) = {$
  underline(
  underline(
    #h(padding) #body #h(padding) #h(offset)
    #text(subsize, baseline: baseline)[#sub]
  ))$
}

#let num2Alphabet(n) = {
  numbering("A", n)
}

#let num2alphabet(n) = {
  numbering("a", n)
}


#show: init
#show: topbar

