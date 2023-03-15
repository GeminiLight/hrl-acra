graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    high 100
    low 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "max_cpu"
    originator "cpu"
    owner "node"
    type "extrema"
  ]
  link_attrs_setting [
    distribution "uniform"
    dtype "int"
    generative 0
    high 100
    low 50
    name "bw"
    owner "link"
    type "resource"
  ]
  link_attrs_setting [
    name "max_bw"
    originator "bw"
    owner "link"
    type "extrema"
  ]
  num_nodes 27
  save_dir "dataset/p_net"
  topology [
    file_path "./dataset/topology/CSTNet.gml"
    type "waxman"
    wm_alpha 0.5
    wm_beta 0.2
  ]
  file_name "p_net.gml"
  node [
    id 0
    label "0"
    cpu 82
    max_cpu 82
  ]
  node [
    id 1
    label "1"
    cpu 99
    max_cpu 99
  ]
  node [
    id 2
    label "2"
    cpu 81
    max_cpu 81
  ]
  node [
    id 3
    label "3"
    cpu 50
    max_cpu 50
  ]
  node [
    id 4
    label "4"
    cpu 94
    max_cpu 94
  ]
  node [
    id 5
    label "5"
    cpu 57
    max_cpu 57
  ]
  node [
    id 6
    label "6"
    cpu 64
    max_cpu 64
  ]
  node [
    id 7
    label "7"
    cpu 92
    max_cpu 92
  ]
  node [
    id 8
    label "8"
    cpu 55
    max_cpu 55
  ]
  node [
    id 9
    label "9"
    cpu 55
    max_cpu 55
  ]
  node [
    id 10
    label "10"
    cpu 82
    max_cpu 82
  ]
  node [
    id 11
    label "11"
    cpu 99
    max_cpu 99
  ]
  node [
    id 12
    label "12"
    cpu 85
    max_cpu 85
  ]
  node [
    id 13
    label "13"
    cpu 62
    max_cpu 62
  ]
  node [
    id 14
    label "14"
    cpu 57
    max_cpu 57
  ]
  node [
    id 15
    label "15"
    cpu 76
    max_cpu 76
  ]
  node [
    id 16
    label "16"
    cpu 63
    max_cpu 63
  ]
  node [
    id 17
    label "17"
    cpu 84
    max_cpu 84
  ]
  node [
    id 18
    label "18"
    cpu 54
    max_cpu 54
  ]
  node [
    id 19
    label "19"
    cpu 54
    max_cpu 54
  ]
  node [
    id 20
    label "20"
    cpu 53
    max_cpu 53
  ]
  node [
    id 21
    label "21"
    cpu 86
    max_cpu 86
  ]
  node [
    id 22
    label "22"
    cpu 84
    max_cpu 84
  ]
  node [
    id 23
    label "23"
    cpu 73
    max_cpu 73
  ]
  node [
    id 24
    label "24"
    cpu 55
    max_cpu 55
  ]
  node [
    id 25
    label "25"
    cpu 64
    max_cpu 64
  ]
  node [
    id 26
    label "26"
    cpu 79
    max_cpu 79
  ]
  edge [
    source 0
    target 1
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 0
    target 2
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 0
    target 3
    color "green"
    bw 10.0
    max_bw 10.0
  ]
  edge [
    source 0
    target 4
    color "green"
    bw 10.0
    max_bw 10.0
  ]
  edge [
    source 0
    target 5
    color "green"
    bw 10.0
    max_bw 10.0
  ]
  edge [
    source 0
    target 6
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 0
    target 8
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 0
    target 10
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 0
    target 12
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 0
    target 13
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 1
    target 3
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 1
    target 14
    color "black"
    bw 1.0
    max_bw 1.0
  ]
  edge [
    source 1
    target 15
    color "yellow"
    bw 2.5
    max_bw 2.5
  ]
  edge [
    source 1
    target 16
    color "black"
    bw 1.0
    max_bw 1.0
  ]
  edge [
    source 1
    target 17
    color "yellow"
    bw 2.5
    max_bw 2.5
  ]
  edge [
    source 1
    target 18
    color "yellow"
    bw 2.5
    max_bw 2.5
  ]
  edge [
    source 1
    target 19
    color "yellow"
    bw 2.5
    max_bw 2.5
  ]
  edge [
    source 1
    target 20
    color "yellow"
    bw 2.5
    max_bw 2.5
  ]
  edge [
    source 2
    target 3
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 2
    target 21
    color "yellow"
    bw 2.5
    max_bw 2.5
  ]
  edge [
    source 2
    target 23
    color "green"
    bw 10.0
    max_bw 10.0
  ]
  edge [
    source 2
    target 24
    color "black"
    bw 1.0
    max_bw 1.0
  ]
  edge [
    source 2
    target 25
    color "yellow"
    bw 2.5
    max_bw 2.5
  ]
  edge [
    source 2
    target 26
    color "black"
    bw 1.0
    max_bw 1.0
  ]
  edge [
    source 3
    target 4
    color "green"
    bw 10.0
    max_bw 10.0
  ]
  edge [
    source 3
    target 5
    color "green"
    bw 10.0
    max_bw 10.0
  ]
  edge [
    source 3
    target 6
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 3
    target 8
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 3
    target 10
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 3
    target 12
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 3
    target 13
    color "red"
    bw 100.0
    max_bw 100.0
  ]
  edge [
    source 6
    target 7
    color "black"
    bw 1.0
    max_bw 1.0
  ]
  edge [
    source 8
    target 9
    color "green"
    bw 10.0
    max_bw 10.0
  ]
  edge [
    source 10
    target 11
    color "green"
    bw 10.0
    max_bw 10.0
  ]
  edge [
    source 19
    target 20
    color "black"
    bw 1.0
    max_bw 1.0
  ]
  edge [
    source 22
    target 23
    color "green"
    bw 10.0
    max_bw 10.0
  ]
]
