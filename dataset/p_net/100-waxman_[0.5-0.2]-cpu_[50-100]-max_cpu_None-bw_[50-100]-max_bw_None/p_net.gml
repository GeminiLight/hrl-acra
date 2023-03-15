graph [
  node_attrs_setting [
    name "cpu"
    owner "node"
    type "resource"
    generative 1
    dtype "int"
    distribution "uniform"
    high 100
    low 50
  ]
  node_attrs_setting [
    name "max_cpu"
    owner "node"
    type "extrema"
    originator "cpu"
  ]
  link_attrs_setting [
    name "bw"
    owner "link"
    type "resource"
    generative 1
    dtype "int"
    distribution "uniform"
    high 100
    low 50
  ]
  link_attrs_setting [
    name "max_bw"
    owner "link"
    type "extrema"
    originator "bw"
  ]
  save_dir "dataset/p_net"
  file_name "p_net.gml"
  num_nodes 100
  type "waxman"
  wm_alpha 0.5
  wm_beta 0.2
  node [
    id 0
    label "0"
    pos 0.34514031705243664
    pos 0.5928961975415388
    cpu 73
    max_cpu 73
  ]
  node [
    id 1
    label "1"
    pos 0.8329282917886225
    pos 0.3532784899063923
    cpu 59
    max_cpu 59
  ]
  node [
    id 2
    label "2"
    pos 0.3686834356119244
    pos 0.6294062826564973
    cpu 94
    max_cpu 94
  ]
  node [
    id 3
    label "3"
    pos 0.3718822275198047
    pos 0.43247907116544015
    cpu 91
    max_cpu 91
  ]
  node [
    id 4
    label "4"
    pos 0.10169118913699071
    pos 0.11975777093156259
    cpu 83
    max_cpu 83
  ]
  node [
    id 5
    label "5"
    pos 0.3709409637846619
    pos 0.17591475354615438
    cpu 60
    max_cpu 60
  ]
  node [
    id 6
    label "6"
    pos 0.8571877144488291
    pos 0.8128159322739145
    cpu 73
    max_cpu 73
  ]
  node [
    id 7
    label "7"
    pos 0.3135769286266654
    pos 0.11723277208259275
    cpu 81
    max_cpu 81
  ]
  node [
    id 8
    label "8"
    pos 0.9731539629093557
    pos 0.14790514998024984
    cpu 56
    max_cpu 56
  ]
  node [
    id 9
    label "9"
    pos 0.4728255116761886
    pos 0.00802919814015468
    cpu 94
    max_cpu 94
  ]
  node [
    id 10
    label "10"
    pos 0.23156783021883587
    pos 0.22010267626566016
    cpu 97
    max_cpu 97
  ]
  node [
    id 11
    label "11"
    pos 0.724954115235689
    pos 0.821738492305897
    cpu 88
    max_cpu 88
  ]
  node [
    id 12
    label "12"
    pos 0.24774807470674198
    pos 0.16869428105885553
    cpu 75
    max_cpu 75
  ]
  node [
    id 13
    label "13"
    pos 0.13616717479723006
    pos 0.11939568330463701
    cpu 62
    max_cpu 62
  ]
  node [
    id 14
    label "14"
    pos 0.628111179238554
    pos 0.05245681999760399
    cpu 87
    max_cpu 87
  ]
  node [
    id 15
    label "15"
    pos 0.04056918064341619
    pos 0.7598231500709008
    cpu 98
    max_cpu 98
  ]
  node [
    id 16
    label "16"
    pos 0.008040588506506796
    pos 0.8544280337594877
    cpu 56
    max_cpu 56
  ]
  node [
    id 17
    label "17"
    pos 0.2532147555711779
    pos 0.17862396954281456
    cpu 92
    max_cpu 92
  ]
  node [
    id 18
    label "18"
    pos 0.2849699751965803
    pos 0.2314169658740517
    cpu 85
    max_cpu 85
  ]
  node [
    id 19
    label "19"
    pos 0.09278531109815713
    pos 0.5690318444529848
    cpu 68
    max_cpu 68
  ]
  node [
    id 20
    label "20"
    pos 0.9063071882926805
    pos 0.5953122884307369
    cpu 62
    max_cpu 62
  ]
  node [
    id 21
    label "21"
    pos 0.33122178235387023
    pos 0.4185375484663858
    cpu 68
    max_cpu 68
  ]
  node [
    id 22
    label "22"
    pos 0.9731089276673117
    pos 0.9844200216206102
    cpu 70
    max_cpu 70
  ]
  node [
    id 23
    label "23"
    pos 0.295002011284377
    pos 0.9330760978253129
    cpu 61
    max_cpu 61
  ]
  node [
    id 24
    label "24"
    pos 0.9093990410247045
    pos 0.5953322951495008
    cpu 56
    max_cpu 56
  ]
  node [
    id 25
    label "25"
    pos 0.9365747370529741
    pos 0.6115438574109544
    cpu 83
    max_cpu 83
  ]
  node [
    id 26
    label "26"
    pos 0.5026186850950412
    pos 0.9401301466266411
    cpu 85
    max_cpu 85
  ]
  node [
    id 27
    label "27"
    pos 0.1292965908984287
    pos 0.9414103344420319
    cpu 65
    max_cpu 65
  ]
  node [
    id 28
    label "28"
    pos 0.4350282900667809
    pos 0.2875920687766603
    cpu 50
    max_cpu 50
  ]
  node [
    id 29
    label "29"
    pos 0.8149844367592624
    pos 0.7285510958159581
    cpu 53
    max_cpu 53
  ]
  node [
    id 30
    label "30"
    pos 0.3655948171540012
    pos 0.9881919448844979
    cpu 55
    max_cpu 55
  ]
  node [
    id 31
    label "31"
    pos 0.3808770401032556
    pos 0.6595601147540561
    cpu 89
    max_cpu 89
  ]
  node [
    id 32
    label "32"
    pos 0.3215125985628353
    pos 0.090113773408935
    cpu 89
    max_cpu 89
  ]
  node [
    id 33
    label "33"
    pos 0.05989064127513599
    pos 0.7458221599509925
    cpu 99
    max_cpu 99
  ]
  node [
    id 34
    label "34"
    pos 0.07221539783373854
    pos 0.3236250927406611
    cpu 71
    max_cpu 71
  ]
  node [
    id 35
    label "35"
    pos 0.3212052185505402
    pos 0.5909536796988334
    cpu 50
    max_cpu 50
  ]
  node [
    id 36
    label "36"
    pos 0.515177421085109
    pos 0.27888209959303956
    cpu 94
    max_cpu 94
  ]
  node [
    id 37
    label "37"
    pos 0.06861891389003472
    pos 0.04421183656179517
    cpu 86
    max_cpu 86
  ]
  node [
    id 38
    label "38"
    pos 0.6517097966697374
    pos 0.8170185407385185
    cpu 76
    max_cpu 76
  ]
  node [
    id 39
    label "39"
    pos 0.8068495280380743
    pos 0.3084531341856046
    cpu 59
    max_cpu 59
  ]
  node [
    id 40
    label "40"
    pos 0.9101179738819678
    pos 0.7715107254696052
    cpu 91
    max_cpu 91
  ]
  node [
    id 41
    label "41"
    pos 0.699210082071327
    pos 0.2403535907314409
    cpu 95
    max_cpu 95
  ]
  node [
    id 42
    label "42"
    pos 0.3972866431770653
    pos 0.19749394097331896
    cpu 88
    max_cpu 88
  ]
  node [
    id 43
    label "43"
    pos 0.741102803281465
    pos 0.19072512761041538
    cpu 88
    max_cpu 88
  ]
  node [
    id 44
    label "44"
    pos 0.451383778249172
    pos 0.4201570395884451
    cpu 75
    max_cpu 75
  ]
  node [
    id 45
    label "45"
    pos 0.3354989099147526
    pos 0.30414918708768723
    cpu 93
    max_cpu 93
  ]
  node [
    id 46
    label "46"
    pos 0.1745483280318929
    pos 0.1249620231402152
    cpu 79
    max_cpu 79
  ]
  node [
    id 47
    label "47"
    pos 0.5433164007803994
    pos 0.21861099098530778
    cpu 65
    max_cpu 65
  ]
  node [
    id 48
    label "48"
    pos 0.7550273708613935
    pos 0.9018018124092386
    cpu 63
    max_cpu 63
  ]
  node [
    id 49
    label "49"
    pos 0.3202124525967812
    pos 0.967777936974122
    cpu 93
    max_cpu 93
  ]
  node [
    id 50
    label "50"
    pos 0.1872783498450311
    pos 0.350753593202179
    cpu 78
    max_cpu 78
  ]
  node [
    id 51
    label "51"
    pos 0.27128813127502815
    pos 0.7518586499608244
    cpu 78
    max_cpu 78
  ]
  node [
    id 52
    label "52"
    pos 0.09958008243490446
    pos 0.9929158149893972
    cpu 87
    max_cpu 87
  ]
  node [
    id 53
    label "53"
    pos 0.21301935090197344
    pos 0.9217048214567771
    cpu 77
    max_cpu 77
  ]
  node [
    id 54
    label "54"
    pos 0.6592251607639817
    pos 0.715841760507337
    cpu 92
    max_cpu 92
  ]
  node [
    id 55
    label "55"
    pos 0.03393094299648247
    pos 0.6995420302079243
    cpu 89
    max_cpu 89
  ]
  node [
    id 56
    label "56"
    pos 0.2504327004890492
    pos 0.25566644137571337
    cpu 74
    max_cpu 74
  ]
  node [
    id 57
    label "57"
    pos 0.20400566207955173
    pos 0.15177050728232655
    cpu 88
    max_cpu 88
  ]
  node [
    id 58
    label "58"
    pos 0.007842561529665337
    pos 0.4277473450404756
    cpu 78
    max_cpu 78
  ]
  node [
    id 59
    label "59"
    pos 0.5228184768124915
    pos 0.5730130082902984
    cpu 55
    max_cpu 55
  ]
  node [
    id 60
    label "60"
    pos 0.5120784687742388
    pos 0.6873137730004539
    cpu 78
    max_cpu 78
  ]
  node [
    id 61
    label "61"
    pos 0.5532107556529949
    pos 0.053665640247570257
    cpu 82
    max_cpu 82
  ]
  node [
    id 62
    label "62"
    pos 0.008535955797582706
    pos 0.5636448875262573
    cpu 59
    max_cpu 59
  ]
  node [
    id 63
    label "63"
    pos 0.5073660757162078
    pos 0.7425883843620444
    cpu 87
    max_cpu 87
  ]
  node [
    id 64
    label "64"
    pos 0.9879420251793994
    pos 0.3083764110753139
    cpu 50
    max_cpu 50
  ]
  node [
    id 65
    label "65"
    pos 0.8734947952032169
    pos 0.05544162439387346
    cpu 67
    max_cpu 67
  ]
  node [
    id 66
    label "66"
    pos 0.18249493483533774
    pos 0.6492196350931833
    cpu 72
    max_cpu 72
  ]
  node [
    id 67
    label "67"
    pos 0.33029523794295834
    pos 0.9139603144306672
    cpu 82
    max_cpu 82
  ]
  node [
    id 68
    label "68"
    pos 0.5200194123157141
    pos 0.7597052416134337
    cpu 72
    max_cpu 72
  ]
  node [
    id 69
    label "69"
    pos 0.8376982889522687
    pos 0.8223035739066014
    cpu 88
    max_cpu 88
  ]
  node [
    id 70
    label "70"
    pos 0.9284515168652379
    pos 0.06435419032482115
    cpu 50
    max_cpu 50
  ]
  node [
    id 71
    label "71"
    pos 0.3013965296173431
    pos 0.39179029865027004
    cpu 63
    max_cpu 63
  ]
  node [
    id 72
    label "72"
    pos 0.7865275286222181
    pos 0.6902973038812095
    cpu 69
    max_cpu 69
  ]
  node [
    id 73
    label "73"
    pos 0.45638683846747474
    pos 0.7738597967307258
    cpu 70
    max_cpu 70
  ]
  node [
    id 74
    label "74"
    pos 0.8897395045992247
    pos 0.8961180187308982
    cpu 58
    max_cpu 58
  ]
  node [
    id 75
    label "75"
    pos 0.8193303370189103
    pos 0.17867362775665008
    cpu 92
    max_cpu 92
  ]
  node [
    id 76
    label "76"
    pos 0.459964852624304
    pos 0.5482732220128235
    cpu 73
    max_cpu 73
  ]
  node [
    id 77
    label "77"
    pos 0.1854017740580317
    pos 0.9279268913732815
    cpu 85
    max_cpu 85
  ]
  node [
    id 78
    label "78"
    pos 0.44673702767237344
    pos 0.9199728670631009
    cpu 64
    max_cpu 64
  ]
  node [
    id 79
    label "79"
    pos 0.7124892982400711
    pos 0.14339254063346696
    cpu 50
    max_cpu 50
  ]
  node [
    id 80
    label "80"
    pos 0.09109164236102107
    pos 0.6347517181793858
    cpu 98
    max_cpu 98
  ]
  node [
    id 81
    label "81"
    pos 0.7451180555484271
    pos 0.8994952157062294
    cpu 59
    max_cpu 59
  ]
  node [
    id 82
    label "82"
    pos 0.47426502064204656
    pos 0.6796671927375519
    cpu 66
    max_cpu 66
  ]
  node [
    id 83
    label "83"
    pos 0.6737858770240824
    pos 0.46199358340591823
    cpu 77
    max_cpu 77
  ]
  node [
    id 84
    label "84"
    pos 0.7517260537811085
    pos 0.44555937733005746
    cpu 58
    max_cpu 58
  ]
  node [
    id 85
    label "85"
    pos 0.7867263611718336
    pos 0.7807556945685209
    cpu 95
    max_cpu 95
  ]
  node [
    id 86
    label "86"
    pos 0.02335430720182652
    pos 0.7708240591239197
    cpu 69
    max_cpu 69
  ]
  node [
    id 87
    label "87"
    pos 0.28093333941662013
    pos 0.08941543585231426
    cpu 91
    max_cpu 91
  ]
  node [
    id 88
    label "88"
    pos 0.9467590453572198
    pos 0.9438898847800488
    cpu 70
    max_cpu 70
  ]
  node [
    id 89
    label "89"
    pos 0.9175649367446407
    pos 0.7129336595521005
    cpu 88
    max_cpu 88
  ]
  node [
    id 90
    label "90"
    pos 0.7431062102507846
    pos 0.9976316523718299
    cpu 90
    max_cpu 90
  ]
  node [
    id 91
    label "91"
    pos 0.8508159528807862
    pos 0.32680198899995183
    cpu 73
    max_cpu 73
  ]
  node [
    id 92
    label "92"
    pos 0.23368529153794826
    pos 0.6862337070701657
    cpu 51
    max_cpu 51
  ]
  node [
    id 93
    label "93"
    pos 0.9482440740026948
    pos 0.521376070210737
    cpu 60
    max_cpu 60
  ]
  node [
    id 94
    label "94"
    pos 0.19674144640695235
    pos 0.7223008899252775
    cpu 63
    max_cpu 63
  ]
  node [
    id 95
    label "95"
    pos 0.277474002826016
    pos 0.7608127891112547
    cpu 78
    max_cpu 78
  ]
  node [
    id 96
    label "96"
    pos 0.9943224480542839
    pos 0.4773135497266423
    cpu 77
    max_cpu 77
  ]
  node [
    id 97
    label "97"
    pos 0.49930324954712835
    pos 0.2566363479454856
    cpu 75
    max_cpu 75
  ]
  node [
    id 98
    label "98"
    pos 0.5968625833039313
    pos 0.027317368584114265
    cpu 66
    max_cpu 66
  ]
  node [
    id 99
    label "99"
    pos 0.42404703339783056
    pos 0.414219370775213
    cpu 76
    max_cpu 76
  ]
  edge [
    source 0
    target 23
    bw 50
    max_bw 50
  ]
  edge [
    source 0
    target 31
    bw 78
    max_bw 78
  ]
  edge [
    source 0
    target 53
    bw 64
    max_bw 64
  ]
  edge [
    source 0
    target 57
    bw 87
    max_bw 87
  ]
  edge [
    source 0
    target 59
    bw 67
    max_bw 67
  ]
  edge [
    source 0
    target 60
    bw 89
    max_bw 89
  ]
  edge [
    source 0
    target 72
    bw 57
    max_bw 57
  ]
  edge [
    source 0
    target 95
    bw 65
    max_bw 65
  ]
  edge [
    source 1
    target 28
    bw 61
    max_bw 61
  ]
  edge [
    source 1
    target 39
    bw 97
    max_bw 97
  ]
  edge [
    source 1
    target 41
    bw 84
    max_bw 84
  ]
  edge [
    source 1
    target 59
    bw 72
    max_bw 72
  ]
  edge [
    source 1
    target 70
    bw 99
    max_bw 99
  ]
  edge [
    source 1
    target 84
    bw 87
    max_bw 87
  ]
  edge [
    source 1
    target 92
    bw 77
    max_bw 77
  ]
  edge [
    source 1
    target 96
    bw 60
    max_bw 60
  ]
  edge [
    source 2
    target 13
    bw 79
    max_bw 79
  ]
  edge [
    source 2
    target 15
    bw 80
    max_bw 80
  ]
  edge [
    source 2
    target 44
    bw 74
    max_bw 74
  ]
  edge [
    source 2
    target 55
    bw 57
    max_bw 57
  ]
  edge [
    source 2
    target 56
    bw 80
    max_bw 80
  ]
  edge [
    source 2
    target 59
    bw 79
    max_bw 79
  ]
  edge [
    source 2
    target 63
    bw 99
    max_bw 99
  ]
  edge [
    source 2
    target 71
    bw 65
    max_bw 65
  ]
  edge [
    source 2
    target 72
    bw 89
    max_bw 89
  ]
  edge [
    source 2
    target 76
    bw 73
    max_bw 73
  ]
  edge [
    source 2
    target 77
    bw 64
    max_bw 64
  ]
  edge [
    source 2
    target 92
    bw 57
    max_bw 57
  ]
  edge [
    source 2
    target 95
    bw 95
    max_bw 95
  ]
  edge [
    source 3
    target 17
    bw 94
    max_bw 94
  ]
  edge [
    source 3
    target 34
    bw 98
    max_bw 98
  ]
  edge [
    source 3
    target 37
    bw 98
    max_bw 98
  ]
  edge [
    source 3
    target 42
    bw 91
    max_bw 91
  ]
  edge [
    source 3
    target 44
    bw 69
    max_bw 69
  ]
  edge [
    source 3
    target 51
    bw 67
    max_bw 67
  ]
  edge [
    source 3
    target 56
    bw 66
    max_bw 66
  ]
  edge [
    source 3
    target 75
    bw 51
    max_bw 51
  ]
  edge [
    source 3
    target 77
    bw 77
    max_bw 77
  ]
  edge [
    source 3
    target 87
    bw 94
    max_bw 94
  ]
  edge [
    source 4
    target 7
    bw 89
    max_bw 89
  ]
  edge [
    source 4
    target 12
    bw 99
    max_bw 99
  ]
  edge [
    source 4
    target 14
    bw 68
    max_bw 68
  ]
  edge [
    source 4
    target 19
    bw 92
    max_bw 92
  ]
  edge [
    source 4
    target 21
    bw 60
    max_bw 60
  ]
  edge [
    source 4
    target 35
    bw 88
    max_bw 88
  ]
  edge [
    source 4
    target 37
    bw 93
    max_bw 93
  ]
  edge [
    source 4
    target 53
    bw 56
    max_bw 56
  ]
  edge [
    source 4
    target 70
    bw 50
    max_bw 50
  ]
  edge [
    source 5
    target 7
    bw 68
    max_bw 68
  ]
  edge [
    source 5
    target 17
    bw 71
    max_bw 71
  ]
  edge [
    source 5
    target 41
    bw 71
    max_bw 71
  ]
  edge [
    source 5
    target 47
    bw 57
    max_bw 57
  ]
  edge [
    source 5
    target 56
    bw 92
    max_bw 92
  ]
  edge [
    source 5
    target 71
    bw 73
    max_bw 73
  ]
  edge [
    source 5
    target 73
    bw 78
    max_bw 78
  ]
  edge [
    source 5
    target 87
    bw 50
    max_bw 50
  ]
  edge [
    source 5
    target 97
    bw 81
    max_bw 81
  ]
  edge [
    source 6
    target 20
    bw 77
    max_bw 77
  ]
  edge [
    source 6
    target 25
    bw 57
    max_bw 57
  ]
  edge [
    source 6
    target 40
    bw 92
    max_bw 92
  ]
  edge [
    source 6
    target 68
    bw 52
    max_bw 52
  ]
  edge [
    source 6
    target 73
    bw 83
    max_bw 83
  ]
  edge [
    source 6
    target 81
    bw 96
    max_bw 96
  ]
  edge [
    source 6
    target 88
    bw 92
    max_bw 92
  ]
  edge [
    source 7
    target 10
    bw 76
    max_bw 76
  ]
  edge [
    source 7
    target 13
    bw 60
    max_bw 60
  ]
  edge [
    source 7
    target 50
    bw 69
    max_bw 69
  ]
  edge [
    source 7
    target 54
    bw 99
    max_bw 99
  ]
  edge [
    source 7
    target 87
    bw 69
    max_bw 69
  ]
  edge [
    source 8
    target 14
    bw 50
    max_bw 50
  ]
  edge [
    source 8
    target 75
    bw 60
    max_bw 60
  ]
  edge [
    source 8
    target 79
    bw 59
    max_bw 59
  ]
  edge [
    source 8
    target 83
    bw 99
    max_bw 99
  ]
  edge [
    source 9
    target 17
    bw 84
    max_bw 84
  ]
  edge [
    source 9
    target 28
    bw 70
    max_bw 70
  ]
  edge [
    source 9
    target 46
    bw 58
    max_bw 58
  ]
  edge [
    source 9
    target 61
    bw 93
    max_bw 93
  ]
  edge [
    source 10
    target 13
    bw 65
    max_bw 65
  ]
  edge [
    source 10
    target 14
    bw 80
    max_bw 80
  ]
  edge [
    source 10
    target 17
    bw 86
    max_bw 86
  ]
  edge [
    source 10
    target 21
    bw 97
    max_bw 97
  ]
  edge [
    source 11
    target 29
    bw 85
    max_bw 85
  ]
  edge [
    source 11
    target 63
    bw 93
    max_bw 93
  ]
  edge [
    source 11
    target 74
    bw 65
    max_bw 65
  ]
  edge [
    source 11
    target 85
    bw 71
    max_bw 71
  ]
  edge [
    source 11
    target 88
    bw 62
    max_bw 62
  ]
  edge [
    source 11
    target 95
    bw 68
    max_bw 68
  ]
  edge [
    source 12
    target 18
    bw 87
    max_bw 87
  ]
  edge [
    source 12
    target 25
    bw 98
    max_bw 98
  ]
  edge [
    source 12
    target 32
    bw 76
    max_bw 76
  ]
  edge [
    source 12
    target 42
    bw 68
    max_bw 68
  ]
  edge [
    source 12
    target 46
    bw 89
    max_bw 89
  ]
  edge [
    source 12
    target 50
    bw 89
    max_bw 89
  ]
  edge [
    source 12
    target 56
    bw 76
    max_bw 76
  ]
  edge [
    source 12
    target 57
    bw 61
    max_bw 61
  ]
  edge [
    source 12
    target 76
    bw 76
    max_bw 76
  ]
  edge [
    source 12
    target 82
    bw 90
    max_bw 90
  ]
  edge [
    source 12
    target 92
    bw 76
    max_bw 76
  ]
  edge [
    source 13
    target 18
    bw 62
    max_bw 62
  ]
  edge [
    source 13
    target 19
    bw 56
    max_bw 56
  ]
  edge [
    source 13
    target 35
    bw 80
    max_bw 80
  ]
  edge [
    source 13
    target 39
    bw 89
    max_bw 89
  ]
  edge [
    source 13
    target 56
    bw 65
    max_bw 65
  ]
  edge [
    source 13
    target 61
    bw 95
    max_bw 95
  ]
  edge [
    source 13
    target 82
    bw 94
    max_bw 94
  ]
  edge [
    source 13
    target 94
    bw 85
    max_bw 85
  ]
  edge [
    source 14
    target 25
    bw 55
    max_bw 55
  ]
  edge [
    source 14
    target 41
    bw 53
    max_bw 53
  ]
  edge [
    source 14
    target 44
    bw 96
    max_bw 96
  ]
  edge [
    source 14
    target 47
    bw 90
    max_bw 90
  ]
  edge [
    source 14
    target 97
    bw 81
    max_bw 81
  ]
  edge [
    source 15
    target 16
    bw 81
    max_bw 81
  ]
  edge [
    source 15
    target 23
    bw 64
    max_bw 64
  ]
  edge [
    source 15
    target 33
    bw 90
    max_bw 90
  ]
  edge [
    source 15
    target 62
    bw 59
    max_bw 59
  ]
  edge [
    source 15
    target 68
    bw 66
    max_bw 66
  ]
  edge [
    source 15
    target 92
    bw 98
    max_bw 98
  ]
  edge [
    source 15
    target 94
    bw 95
    max_bw 95
  ]
  edge [
    source 16
    target 19
    bw 67
    max_bw 67
  ]
  edge [
    source 16
    target 30
    bw 84
    max_bw 84
  ]
  edge [
    source 16
    target 49
    bw 71
    max_bw 71
  ]
  edge [
    source 16
    target 67
    bw 51
    max_bw 51
  ]
  edge [
    source 16
    target 73
    bw 54
    max_bw 54
  ]
  edge [
    source 16
    target 80
    bw 74
    max_bw 74
  ]
  edge [
    source 16
    target 95
    bw 92
    max_bw 92
  ]
  edge [
    source 17
    target 18
    bw 64
    max_bw 64
  ]
  edge [
    source 17
    target 21
    bw 97
    max_bw 97
  ]
  edge [
    source 17
    target 33
    bw 66
    max_bw 66
  ]
  edge [
    source 17
    target 44
    bw 62
    max_bw 62
  ]
  edge [
    source 17
    target 58
    bw 90
    max_bw 90
  ]
  edge [
    source 17
    target 76
    bw 66
    max_bw 66
  ]
  edge [
    source 17
    target 99
    bw 55
    max_bw 55
  ]
  edge [
    source 18
    target 23
    bw 91
    max_bw 91
  ]
  edge [
    source 18
    target 29
    bw 57
    max_bw 57
  ]
  edge [
    source 18
    target 32
    bw 78
    max_bw 78
  ]
  edge [
    source 18
    target 42
    bw 79
    max_bw 79
  ]
  edge [
    source 18
    target 45
    bw 87
    max_bw 87
  ]
  edge [
    source 18
    target 47
    bw 66
    max_bw 66
  ]
  edge [
    source 18
    target 50
    bw 84
    max_bw 84
  ]
  edge [
    source 18
    target 56
    bw 59
    max_bw 59
  ]
  edge [
    source 18
    target 62
    bw 60
    max_bw 60
  ]
  edge [
    source 18
    target 77
    bw 55
    max_bw 55
  ]
  edge [
    source 18
    target 94
    bw 83
    max_bw 83
  ]
  edge [
    source 18
    target 99
    bw 77
    max_bw 77
  ]
  edge [
    source 19
    target 31
    bw 81
    max_bw 81
  ]
  edge [
    source 19
    target 34
    bw 50
    max_bw 50
  ]
  edge [
    source 19
    target 37
    bw 65
    max_bw 65
  ]
  edge [
    source 19
    target 38
    bw 65
    max_bw 65
  ]
  edge [
    source 19
    target 51
    bw 95
    max_bw 95
  ]
  edge [
    source 19
    target 57
    bw 65
    max_bw 65
  ]
  edge [
    source 19
    target 58
    bw 61
    max_bw 61
  ]
  edge [
    source 19
    target 80
    bw 99
    max_bw 99
  ]
  edge [
    source 19
    target 84
    bw 87
    max_bw 87
  ]
  edge [
    source 19
    target 92
    bw 61
    max_bw 61
  ]
  edge [
    source 19
    target 94
    bw 79
    max_bw 79
  ]
  edge [
    source 20
    target 24
    bw 82
    max_bw 82
  ]
  edge [
    source 20
    target 26
    bw 82
    max_bw 82
  ]
  edge [
    source 20
    target 33
    bw 87
    max_bw 87
  ]
  edge [
    source 20
    target 83
    bw 91
    max_bw 91
  ]
  edge [
    source 21
    target 28
    bw 53
    max_bw 53
  ]
  edge [
    source 21
    target 36
    bw 68
    max_bw 68
  ]
  edge [
    source 21
    target 62
    bw 54
    max_bw 54
  ]
  edge [
    source 21
    target 69
    bw 51
    max_bw 51
  ]
  edge [
    source 21
    target 84
    bw 79
    max_bw 79
  ]
  edge [
    source 21
    target 98
    bw 58
    max_bw 58
  ]
  edge [
    source 22
    target 29
    bw 98
    max_bw 98
  ]
  edge [
    source 22
    target 69
    bw 68
    max_bw 68
  ]
  edge [
    source 22
    target 71
    bw 73
    max_bw 73
  ]
  edge [
    source 22
    target 75
    bw 55
    max_bw 55
  ]
  edge [
    source 22
    target 77
    bw 58
    max_bw 58
  ]
  edge [
    source 22
    target 89
    bw 73
    max_bw 73
  ]
  edge [
    source 23
    target 48
    bw 66
    max_bw 66
  ]
  edge [
    source 23
    target 67
    bw 87
    max_bw 87
  ]
  edge [
    source 23
    target 76
    bw 62
    max_bw 62
  ]
  edge [
    source 23
    target 78
    bw 52
    max_bw 52
  ]
  edge [
    source 23
    target 88
    bw 75
    max_bw 75
  ]
  edge [
    source 24
    target 26
    bw 95
    max_bw 95
  ]
  edge [
    source 24
    target 34
    bw 75
    max_bw 75
  ]
  edge [
    source 24
    target 39
    bw 65
    max_bw 65
  ]
  edge [
    source 24
    target 45
    bw 80
    max_bw 80
  ]
  edge [
    source 24
    target 88
    bw 50
    max_bw 50
  ]
  edge [
    source 25
    target 47
    bw 82
    max_bw 82
  ]
  edge [
    source 25
    target 64
    bw 80
    max_bw 80
  ]
  edge [
    source 25
    target 71
    bw 74
    max_bw 74
  ]
  edge [
    source 25
    target 83
    bw 74
    max_bw 74
  ]
  edge [
    source 25
    target 91
    bw 72
    max_bw 72
  ]
  edge [
    source 26
    target 29
    bw 73
    max_bw 73
  ]
  edge [
    source 26
    target 31
    bw 89
    max_bw 89
  ]
  edge [
    source 26
    target 52
    bw 73
    max_bw 73
  ]
  edge [
    source 26
    target 78
    bw 99
    max_bw 99
  ]
  edge [
    source 26
    target 81
    bw 77
    max_bw 77
  ]
  edge [
    source 26
    target 85
    bw 96
    max_bw 96
  ]
  edge [
    source 27
    target 28
    bw 60
    max_bw 60
  ]
  edge [
    source 27
    target 55
    bw 71
    max_bw 71
  ]
  edge [
    source 27
    target 59
    bw 51
    max_bw 51
  ]
  edge [
    source 27
    target 63
    bw 77
    max_bw 77
  ]
  edge [
    source 27
    target 68
    bw 74
    max_bw 74
  ]
  edge [
    source 27
    target 86
    bw 97
    max_bw 97
  ]
  edge [
    source 28
    target 32
    bw 94
    max_bw 94
  ]
  edge [
    source 28
    target 41
    bw 71
    max_bw 71
  ]
  edge [
    source 28
    target 45
    bw 99
    max_bw 99
  ]
  edge [
    source 28
    target 56
    bw 92
    max_bw 92
  ]
  edge [
    source 28
    target 79
    bw 64
    max_bw 64
  ]
  edge [
    source 29
    target 40
    bw 75
    max_bw 75
  ]
  edge [
    source 29
    target 41
    bw 82
    max_bw 82
  ]
  edge [
    source 29
    target 49
    bw 88
    max_bw 88
  ]
  edge [
    source 29
    target 69
    bw 73
    max_bw 73
  ]
  edge [
    source 29
    target 76
    bw 50
    max_bw 50
  ]
  edge [
    source 29
    target 85
    bw 52
    max_bw 52
  ]
  edge [
    source 29
    target 88
    bw 61
    max_bw 61
  ]
  edge [
    source 29
    target 93
    bw 61
    max_bw 61
  ]
  edge [
    source 30
    target 31
    bw 94
    max_bw 94
  ]
  edge [
    source 30
    target 53
    bw 87
    max_bw 87
  ]
  edge [
    source 30
    target 54
    bw 92
    max_bw 92
  ]
  edge [
    source 30
    target 63
    bw 75
    max_bw 75
  ]
  edge [
    source 30
    target 67
    bw 91
    max_bw 91
  ]
  edge [
    source 30
    target 78
    bw 53
    max_bw 53
  ]
  edge [
    source 31
    target 32
    bw 70
    max_bw 70
  ]
  edge [
    source 31
    target 42
    bw 54
    max_bw 54
  ]
  edge [
    source 31
    target 43
    bw 57
    max_bw 57
  ]
  edge [
    source 31
    target 44
    bw 82
    max_bw 82
  ]
  edge [
    source 31
    target 47
    bw 60
    max_bw 60
  ]
  edge [
    source 31
    target 66
    bw 61
    max_bw 61
  ]
  edge [
    source 31
    target 68
    bw 60
    max_bw 60
  ]
  edge [
    source 31
    target 76
    bw 55
    max_bw 55
  ]
  edge [
    source 32
    target 42
    bw 64
    max_bw 64
  ]
  edge [
    source 32
    target 57
    bw 84
    max_bw 84
  ]
  edge [
    source 32
    target 61
    bw 89
    max_bw 89
  ]
  edge [
    source 33
    target 37
    bw 71
    max_bw 71
  ]
  edge [
    source 33
    target 51
    bw 99
    max_bw 99
  ]
  edge [
    source 33
    target 60
    bw 88
    max_bw 88
  ]
  edge [
    source 33
    target 62
    bw 79
    max_bw 79
  ]
  edge [
    source 33
    target 71
    bw 80
    max_bw 80
  ]
  edge [
    source 33
    target 81
    bw 80
    max_bw 80
  ]
  edge [
    source 33
    target 86
    bw 65
    max_bw 65
  ]
  edge [
    source 33
    target 92
    bw 53
    max_bw 53
  ]
  edge [
    source 33
    target 94
    bw 78
    max_bw 78
  ]
  edge [
    source 33
    target 95
    bw 64
    max_bw 64
  ]
  edge [
    source 34
    target 46
    bw 69
    max_bw 69
  ]
  edge [
    source 34
    target 83
    bw 53
    max_bw 53
  ]
  edge [
    source 35
    target 41
    bw 61
    max_bw 61
  ]
  edge [
    source 35
    target 50
    bw 56
    max_bw 56
  ]
  edge [
    source 35
    target 57
    bw 58
    max_bw 58
  ]
  edge [
    source 35
    target 74
    bw 84
    max_bw 84
  ]
  edge [
    source 35
    target 76
    bw 85
    max_bw 85
  ]
  edge [
    source 35
    target 80
    bw 51
    max_bw 51
  ]
  edge [
    source 35
    target 92
    bw 81
    max_bw 81
  ]
  edge [
    source 35
    target 95
    bw 73
    max_bw 73
  ]
  edge [
    source 36
    target 41
    bw 63
    max_bw 63
  ]
  edge [
    source 36
    target 44
    bw 68
    max_bw 68
  ]
  edge [
    source 36
    target 45
    bw 69
    max_bw 69
  ]
  edge [
    source 36
    target 51
    bw 65
    max_bw 65
  ]
  edge [
    source 36
    target 57
    bw 69
    max_bw 69
  ]
  edge [
    source 36
    target 68
    bw 70
    max_bw 70
  ]
  edge [
    source 36
    target 79
    bw 73
    max_bw 73
  ]
  edge [
    source 36
    target 97
    bw 58
    max_bw 58
  ]
  edge [
    source 37
    target 57
    bw 92
    max_bw 92
  ]
  edge [
    source 37
    target 59
    bw 83
    max_bw 83
  ]
  edge [
    source 38
    target 40
    bw 77
    max_bw 77
  ]
  edge [
    source 38
    target 63
    bw 78
    max_bw 78
  ]
  edge [
    source 38
    target 69
    bw 83
    max_bw 83
  ]
  edge [
    source 38
    target 72
    bw 96
    max_bw 96
  ]
  edge [
    source 38
    target 89
    bw 99
    max_bw 99
  ]
  edge [
    source 39
    target 41
    bw 80
    max_bw 80
  ]
  edge [
    source 39
    target 44
    bw 91
    max_bw 91
  ]
  edge [
    source 39
    target 45
    bw 76
    max_bw 76
  ]
  edge [
    source 39
    target 70
    bw 91
    max_bw 91
  ]
  edge [
    source 39
    target 71
    bw 76
    max_bw 76
  ]
  edge [
    source 39
    target 79
    bw 68
    max_bw 68
  ]
  edge [
    source 39
    target 91
    bw 65
    max_bw 65
  ]
  edge [
    source 40
    target 61
    bw 69
    max_bw 69
  ]
  edge [
    source 40
    target 65
    bw 60
    max_bw 60
  ]
  edge [
    source 40
    target 69
    bw 68
    max_bw 68
  ]
  edge [
    source 40
    target 72
    bw 68
    max_bw 68
  ]
  edge [
    source 40
    target 82
    bw 54
    max_bw 54
  ]
  edge [
    source 40
    target 85
    bw 79
    max_bw 79
  ]
  edge [
    source 40
    target 89
    bw 86
    max_bw 86
  ]
  edge [
    source 40
    target 93
    bw 50
    max_bw 50
  ]
  edge [
    source 41
    target 43
    bw 68
    max_bw 68
  ]
  edge [
    source 41
    target 45
    bw 60
    max_bw 60
  ]
  edge [
    source 41
    target 56
    bw 90
    max_bw 90
  ]
  edge [
    source 41
    target 60
    bw 75
    max_bw 75
  ]
  edge [
    source 41
    target 64
    bw 73
    max_bw 73
  ]
  edge [
    source 41
    target 71
    bw 84
    max_bw 84
  ]
  edge [
    source 41
    target 80
    bw 76
    max_bw 76
  ]
  edge [
    source 41
    target 83
    bw 79
    max_bw 79
  ]
  edge [
    source 41
    target 85
    bw 50
    max_bw 50
  ]
  edge [
    source 42
    target 44
    bw 60
    max_bw 60
  ]
  edge [
    source 42
    target 45
    bw 93
    max_bw 93
  ]
  edge [
    source 42
    target 85
    bw 96
    max_bw 96
  ]
  edge [
    source 42
    target 97
    bw 76
    max_bw 76
  ]
  edge [
    source 43
    target 47
    bw 71
    max_bw 71
  ]
  edge [
    source 43
    target 60
    bw 89
    max_bw 89
  ]
  edge [
    source 43
    target 62
    bw 51
    max_bw 51
  ]
  edge [
    source 43
    target 73
    bw 58
    max_bw 58
  ]
  edge [
    source 43
    target 75
    bw 82
    max_bw 82
  ]
  edge [
    source 43
    target 79
    bw 82
    max_bw 82
  ]
  edge [
    source 43
    target 91
    bw 92
    max_bw 92
  ]
  edge [
    source 43
    target 93
    bw 55
    max_bw 55
  ]
  edge [
    source 44
    target 54
    bw 88
    max_bw 88
  ]
  edge [
    source 44
    target 56
    bw 95
    max_bw 95
  ]
  edge [
    source 44
    target 73
    bw 55
    max_bw 55
  ]
  edge [
    source 44
    target 90
    bw 78
    max_bw 78
  ]
  edge [
    source 44
    target 92
    bw 81
    max_bw 81
  ]
  edge [
    source 44
    target 99
    bw 67
    max_bw 67
  ]
  edge [
    source 45
    target 53
    bw 66
    max_bw 66
  ]
  edge [
    source 45
    target 67
    bw 77
    max_bw 77
  ]
  edge [
    source 45
    target 76
    bw 62
    max_bw 62
  ]
  edge [
    source 45
    target 97
    bw 63
    max_bw 63
  ]
  edge [
    source 45
    target 99
    bw 83
    max_bw 83
  ]
  edge [
    source 47
    target 58
    bw 72
    max_bw 72
  ]
  edge [
    source 47
    target 59
    bw 77
    max_bw 77
  ]
  edge [
    source 47
    target 64
    bw 93
    max_bw 93
  ]
  edge [
    source 47
    target 79
    bw 54
    max_bw 54
  ]
  edge [
    source 47
    target 83
    bw 92
    max_bw 92
  ]
  edge [
    source 47
    target 96
    bw 75
    max_bw 75
  ]
  edge [
    source 47
    target 98
    bw 60
    max_bw 60
  ]
  edge [
    source 48
    target 63
    bw 91
    max_bw 91
  ]
  edge [
    source 48
    target 75
    bw 71
    max_bw 71
  ]
  edge [
    source 48
    target 81
    bw 90
    max_bw 90
  ]
  edge [
    source 48
    target 91
    bw 95
    max_bw 95
  ]
  edge [
    source 48
    target 93
    bw 82
    max_bw 82
  ]
  edge [
    source 48
    target 94
    bw 71
    max_bw 71
  ]
  edge [
    source 48
    target 96
    bw 68
    max_bw 68
  ]
  edge [
    source 49
    target 60
    bw 88
    max_bw 88
  ]
  edge [
    source 49
    target 71
    bw 50
    max_bw 50
  ]
  edge [
    source 49
    target 82
    bw 66
    max_bw 66
  ]
  edge [
    source 50
    target 51
    bw 50
    max_bw 50
  ]
  edge [
    source 50
    target 57
    bw 62
    max_bw 62
  ]
  edge [
    source 50
    target 62
    bw 87
    max_bw 87
  ]
  edge [
    source 50
    target 71
    bw 95
    max_bw 95
  ]
  edge [
    source 50
    target 80
    bw 81
    max_bw 81
  ]
  edge [
    source 50
    target 87
    bw 99
    max_bw 99
  ]
  edge [
    source 51
    target 53
    bw 84
    max_bw 84
  ]
  edge [
    source 51
    target 58
    bw 99
    max_bw 99
  ]
  edge [
    source 51
    target 60
    bw 95
    max_bw 95
  ]
  edge [
    source 51
    target 66
    bw 90
    max_bw 90
  ]
  edge [
    source 51
    target 67
    bw 57
    max_bw 57
  ]
  edge [
    source 51
    target 76
    bw 86
    max_bw 86
  ]
  edge [
    source 51
    target 92
    bw 78
    max_bw 78
  ]
  edge [
    source 51
    target 94
    bw 59
    max_bw 59
  ]
  edge [
    source 51
    target 95
    bw 90
    max_bw 90
  ]
  edge [
    source 52
    target 62
    bw 74
    max_bw 74
  ]
  edge [
    source 52
    target 82
    bw 50
    max_bw 50
  ]
  edge [
    source 52
    target 95
    bw 63
    max_bw 63
  ]
  edge [
    source 53
    target 77
    bw 59
    max_bw 59
  ]
  edge [
    source 53
    target 87
    bw 70
    max_bw 70
  ]
  edge [
    source 54
    target 63
    bw 99
    max_bw 99
  ]
  edge [
    source 54
    target 69
    bw 72
    max_bw 72
  ]
  edge [
    source 54
    target 77
    bw 97
    max_bw 97
  ]
  edge [
    source 54
    target 85
    bw 85
    max_bw 85
  ]
  edge [
    source 54
    target 90
    bw 51
    max_bw 51
  ]
  edge [
    source 54
    target 95
    bw 55
    max_bw 55
  ]
  edge [
    source 55
    target 68
    bw 63
    max_bw 63
  ]
  edge [
    source 55
    target 71
    bw 80
    max_bw 80
  ]
  edge [
    source 55
    target 75
    bw 68
    max_bw 68
  ]
  edge [
    source 55
    target 79
    bw 82
    max_bw 82
  ]
  edge [
    source 55
    target 84
    bw 63
    max_bw 63
  ]
  edge [
    source 55
    target 86
    bw 51
    max_bw 51
  ]
  edge [
    source 55
    target 94
    bw 88
    max_bw 88
  ]
  edge [
    source 56
    target 57
    bw 69
    max_bw 69
  ]
  edge [
    source 56
    target 84
    bw 74
    max_bw 74
  ]
  edge [
    source 56
    target 87
    bw 90
    max_bw 90
  ]
  edge [
    source 56
    target 98
    bw 53
    max_bw 53
  ]
  edge [
    source 57
    target 62
    bw 78
    max_bw 78
  ]
  edge [
    source 57
    target 94
    bw 98
    max_bw 98
  ]
  edge [
    source 57
    target 97
    bw 71
    max_bw 71
  ]
  edge [
    source 58
    target 60
    bw 59
    max_bw 59
  ]
  edge [
    source 58
    target 62
    bw 85
    max_bw 85
  ]
  edge [
    source 58
    target 72
    bw 51
    max_bw 51
  ]
  edge [
    source 58
    target 77
    bw 64
    max_bw 64
  ]
  edge [
    source 58
    target 84
    bw 79
    max_bw 79
  ]
  edge [
    source 58
    target 95
    bw 63
    max_bw 63
  ]
  edge [
    source 59
    target 77
    bw 68
    max_bw 68
  ]
  edge [
    source 59
    target 84
    bw 72
    max_bw 72
  ]
  edge [
    source 60
    target 67
    bw 67
    max_bw 67
  ]
  edge [
    source 60
    target 68
    bw 77
    max_bw 77
  ]
  edge [
    source 60
    target 76
    bw 77
    max_bw 77
  ]
  edge [
    source 60
    target 79
    bw 57
    max_bw 57
  ]
  edge [
    source 60
    target 80
    bw 74
    max_bw 74
  ]
  edge [
    source 60
    target 82
    bw 92
    max_bw 92
  ]
  edge [
    source 60
    target 90
    bw 54
    max_bw 54
  ]
  edge [
    source 60
    target 92
    bw 56
    max_bw 56
  ]
  edge [
    source 60
    target 95
    bw 53
    max_bw 53
  ]
  edge [
    source 61
    target 87
    bw 55
    max_bw 55
  ]
  edge [
    source 62
    target 80
    bw 72
    max_bw 72
  ]
  edge [
    source 62
    target 85
    bw 84
    max_bw 84
  ]
  edge [
    source 62
    target 94
    bw 83
    max_bw 83
  ]
  edge [
    source 63
    target 64
    bw 97
    max_bw 97
  ]
  edge [
    source 63
    target 67
    bw 91
    max_bw 91
  ]
  edge [
    source 63
    target 68
    bw 79
    max_bw 79
  ]
  edge [
    source 63
    target 73
    bw 64
    max_bw 64
  ]
  edge [
    source 63
    target 77
    bw 82
    max_bw 82
  ]
  edge [
    source 63
    target 89
    bw 54
    max_bw 54
  ]
  edge [
    source 63
    target 94
    bw 63
    max_bw 63
  ]
  edge [
    source 64
    target 65
    bw 71
    max_bw 71
  ]
  edge [
    source 64
    target 73
    bw 63
    max_bw 63
  ]
  edge [
    source 64
    target 85
    bw 54
    max_bw 54
  ]
  edge [
    source 64
    target 91
    bw 97
    max_bw 97
  ]
  edge [
    source 64
    target 97
    bw 90
    max_bw 90
  ]
  edge [
    source 64
    target 98
    bw 65
    max_bw 65
  ]
  edge [
    source 65
    target 70
    bw 70
    max_bw 70
  ]
  edge [
    source 65
    target 75
    bw 77
    max_bw 77
  ]
  edge [
    source 65
    target 91
    bw 52
    max_bw 52
  ]
  edge [
    source 66
    target 80
    bw 90
    max_bw 90
  ]
  edge [
    source 66
    target 86
    bw 85
    max_bw 85
  ]
  edge [
    source 66
    target 94
    bw 80
    max_bw 80
  ]
  edge [
    source 67
    target 74
    bw 70
    max_bw 70
  ]
  edge [
    source 68
    target 78
    bw 72
    max_bw 72
  ]
  edge [
    source 68
    target 80
    bw 88
    max_bw 88
  ]
  edge [
    source 69
    target 84
    bw 54
    max_bw 54
  ]
  edge [
    source 69
    target 92
    bw 57
    max_bw 57
  ]
  edge [
    source 70
    target 75
    bw 70
    max_bw 70
  ]
  edge [
    source 70
    target 91
    bw 87
    max_bw 87
  ]
  edge [
    source 70
    target 98
    bw 71
    max_bw 71
  ]
  edge [
    source 71
    target 76
    bw 51
    max_bw 51
  ]
  edge [
    source 71
    target 79
    bw 90
    max_bw 90
  ]
  edge [
    source 71
    target 94
    bw 52
    max_bw 52
  ]
  edge [
    source 72
    target 89
    bw 85
    max_bw 85
  ]
  edge [
    source 73
    target 76
    bw 94
    max_bw 94
  ]
  edge [
    source 73
    target 82
    bw 58
    max_bw 58
  ]
  edge [
    source 73
    target 83
    bw 54
    max_bw 54
  ]
  edge [
    source 73
    target 89
    bw 73
    max_bw 73
  ]
  edge [
    source 74
    target 88
    bw 57
    max_bw 57
  ]
  edge [
    source 75
    target 79
    bw 67
    max_bw 67
  ]
  edge [
    source 75
    target 84
    bw 81
    max_bw 81
  ]
  edge [
    source 75
    target 87
    bw 91
    max_bw 91
  ]
  edge [
    source 75
    target 91
    bw 53
    max_bw 53
  ]
  edge [
    source 75
    target 99
    bw 86
    max_bw 86
  ]
  edge [
    source 76
    target 83
    bw 52
    max_bw 52
  ]
  edge [
    source 76
    target 85
    bw 53
    max_bw 53
  ]
  edge [
    source 76
    target 89
    bw 72
    max_bw 72
  ]
  edge [
    source 77
    target 94
    bw 99
    max_bw 99
  ]
  edge [
    source 78
    target 83
    bw 86
    max_bw 86
  ]
  edge [
    source 78
    target 95
    bw 97
    max_bw 97
  ]
  edge [
    source 80
    target 92
    bw 75
    max_bw 75
  ]
  edge [
    source 81
    target 86
    bw 70
    max_bw 70
  ]
  edge [
    source 81
    target 88
    bw 59
    max_bw 59
  ]
  edge [
    source 81
    target 89
    bw 83
    max_bw 83
  ]
  edge [
    source 81
    target 90
    bw 69
    max_bw 69
  ]
  edge [
    source 81
    target 99
    bw 94
    max_bw 94
  ]
  edge [
    source 82
    target 95
    bw 77
    max_bw 77
  ]
  edge [
    source 83
    target 85
    bw 98
    max_bw 98
  ]
  edge [
    source 84
    target 93
    bw 71
    max_bw 71
  ]
  edge [
    source 84
    target 96
    bw 57
    max_bw 57
  ]
  edge [
    source 85
    target 96
    bw 91
    max_bw 91
  ]
  edge [
    source 86
    target 90
    bw 93
    max_bw 93
  ]
  edge [
    source 86
    target 92
    bw 82
    max_bw 82
  ]
  edge [
    source 88
    target 90
    bw 56
    max_bw 56
  ]
  edge [
    source 88
    target 93
    bw 78
    max_bw 78
  ]
  edge [
    source 89
    target 91
    bw 51
    max_bw 51
  ]
  edge [
    source 89
    target 96
    bw 90
    max_bw 90
  ]
  edge [
    source 89
    target 99
    bw 87
    max_bw 87
  ]
  edge [
    source 91
    target 98
    bw 93
    max_bw 93
  ]
  edge [
    source 92
    target 95
    bw 69
    max_bw 69
  ]
  edge [
    source 95
    target 97
    bw 76
    max_bw 76
  ]
  edge [
    source 97
    target 99
    bw 66
    max_bw 66
  ]
  edge [
    source 98
    target 99
    bw 76
    max_bw 76
  ]
]
