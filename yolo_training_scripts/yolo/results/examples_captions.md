# Example captions (paste under the pictures in the README)

- `results/correct/example_1.jpg` (20261002_145203_04b7_primary.rf.67e445ad0aa8158577b0fa0ad57b8eb9.jpg): every object found with the exact right class
- `results/correct/example_2.jpg` (20260925_141331_b1eb_primary.rf.66b78ad3c521fefc76e31a52dad2831f.jpg): every object found with the exact right class
- `results/wrong_object/example_1.jpg` (20260930_161426_5a08_primary.rf.815a0a2d4293ffa0495f7727f0b040e0.jpg): true **precision screwdriver / Metal** -> predicted **scissors -  Material  plastic** (0.52)
- `results/wrong_object/example_2.jpg` (20260918_163155_a198_primary.rf.36eb082d409680fc03dd1b2acb5db28e.jpg): true **scissors -  Material metal** -> predicted **retractable gel pen / plastic- Pdfname  asdasdaswe** (0.44)
- `results/missed_no_detection/example_1.jpg` (20260911_143517_e5a6_primary_split_r.rf.a3b71581a6ed0672f0bcbe13664133dc.jpg): true **precision screwdriver / rubber / black and orange** -> predicted **-**
- `results/missed_no_detection/example_2.jpg` (20260914_170402_d0c9_primary.rf.54507297e25dd9feff13140f3ab84e76.jpg): true **desktop tape dispenser /  plastic tape** -> predicted **-**
- `results/wrong_material_and_color/example_1.jpg` (20260918_173342_ab51_primary.rf.b35fb71e56c40f0043abac4912d71c74.jpg): true **cardboard shipping box-  Material plastic  /  blue -Pdfname OHMlPZx8zUzd** -> predicted **cardboard shipping box / composite -  Color brown** (0.82)
- `results/wrong_material_and_color/example_2.jpg` (20260911_145315_ef6e_primary.rf.d35736d28a3eaaad31460071b138a87e.jpg): true **robotic gripper finger assembly / plastic / blue** -> predicted **robotic gripper finger assembly  / metal -  Pdfname Yy6dySnH33wN** (0.53)
- `results/wrong_material/example_1.jpg` (20261002_142159_8032_primary.rf.56dc2fc257202c26890c77484507e062.jpg): true **protective phone case /  plastic - pdfname asdfeafgta** -> predicted **protective phone case /   rubber  - pdfname asdfeafgta** (0.54)
- `results/wrong_material/example_2.jpg` (20260921_145959_0b29_primary.rf.f4a6b4977846bf1cfc66217ab824ea15.jpg): true **wire spool  -Material plastic -  Color oranges** -> predicted **wire spool /  rubber / oranges -  Pdfname  ZryFz3DBSiO8** (0.50)
- `results/poor_localization/example_1.jpg` (20260925_172522_b6cf_primary.rf.08b05dfba741fd202f93a462d6f59f9a.jpg): true **graphing calculator slide case / plastic** -> predicted **graphing calculator slide case / plastic** (0.91)
- `results/poor_localization/example_2.jpg` (20260925_154108_28aa_primary.rf.cb98a1f533c262d6a8c4934cbe888a18.jpg): true **claw hammer -Material metal** -> predicted **claw hammer -Material metal** (0.95)
- `results/missed_low_confidence/example_1.jpg` (20260925_162626_c12a_primary.rf.e23bdd095b5a57a67774ec415716c44f.jpg): true **robotic gripper finger assembly  / metal -  Pdfname Yy6dySnH33wN** -> predicted **robotic gripper finger assembly  / metal -  Pdfname Yy6dySnH33wN** (0.06)
- `results/missed_low_confidence/example_2.jpg` (20260918_151019_0cf8_primary.rf.5d16b94a18d4eabec744fd90388e8a0c.jpg): true **ballpoint pen / rubber  / red** -> predicted **ballpoint pen / rubber  / red** (0.30)
- `results/false_positive_background/example_1.jpg` (20260925_142407_c4f5_primary.rf.e1e78b784962694b8abf3c3de9819deb.jpg): true **-** -> predicted **desktop tape dispenser / plastic  ** (0.60)
- `results/false_positive_background/example_2.jpg` (20260911_145111_d6ff_primary.rf.6b23587cdc39142be6ac262c9b74015d.jpg): true **-** -> predicted **- extruded aluminum profile -  Material metal** (0.38)
- `results/false_positive_duplicate/example_1.jpg` (20260914_165221_49e5_primary.rf.9b9af7ec04dad3110fea0d42b9ceed48.jpg): true **-** -> predicted **wire spool  -Material plastic -  Color oranges** (0.84)
- `results/false_positive_duplicate/example_2.jpg` (20260911_143556_b83a_primary_split_g.rf.acd9dbdb17a72c55bc3e849cfb85a213.jpg): true **-** -> predicted **ballpoint pen / rubber  / red** (0.31)
