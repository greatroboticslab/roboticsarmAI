# Example captions (paste under the pictures in the README)

- `results/correct/example_1.jpg` (20260930_164241_6104_primary.rf.c4e908e744a749558d2a1244f74a71e6.jpg): every object found with the exact right class
- `results/correct/example_2.jpg` (20260921_152623_4f1d_primary.rf.37b2fb27de2ca956897ba3b1aa8a1f4f.jpg): every object found with the exact right class
- `results/wrong_object/example_1.jpg` (20260930_155406_7b1a_primary.rf.8d25c60ad9d6be219a297d4eea5ee0c0.jpg): true **precision screwdriver / rubber / black and orange** -> predicted **rubber duck -  Material rubber  -Color blue** (0.31)
- `results/wrong_object/example_2.jpg` (20260911_150837_b9b2_primary_split_r.rf.5bfd415b21c17658245883460ccdd075.jpg): true **wireless earbud charging case / plastic** -> predicted **plastic container lid / plastic / red -  Pdfname  ZC63DBkv6oJY** (0.54)
- `results/missed_no_detection/example_1.jpg` (20260911_144838_93de_primary_split_r.rf.d5c08b11118743f9e52f26aaaace971e.jpg): true **car key fob with keyring / metal ** -> predicted **-**
- `results/missed_no_detection/example_2.jpg` (20260918_154445_d571_primary.rf.51184e95d1f3ef87e14acc8243c43294.jpg): true **retractable gel pen / metal** -> predicted **-**
- `results/wrong_material_and_color/example_1.jpg` (20260918_171600_2a5e_primary.rf.29b3ab724d57efcd2c60d2fde9534b8a.jpg): true **cardboard shipping box-  Material plastic  /  blue -Pdfname OHMlPZx8zUzd** -> predicted **cardboard shipping box / composite -  Color brown** (0.66)
- `results/wrong_material_and_color/example_2.jpg` (20260925_162626_c12a_primary.rf.e23bdd095b5a57a67774ec415716c44f.jpg): true **robotic gripper finger assembly  / metal -  Pdfname Yy6dySnH33wN** -> predicted **robotic gripper finger assembly / plastic / blue** (0.79)
- `results/wrong_material/example_1.jpg` (20260921_145828_b766_primary.rf.b77a3399363313cdcf5e1b8341984b44.jpg): true **wire spool  -Material plastic -  Color oranges** -> predicted **wire spool /  rubber / oranges -  Pdfname  ZryFz3DBSiO8** (0.51)
- `results/wrong_material/example_2.jpg` (20260921_145920_256d_primary.rf.07c861ff74fabce1db89b0d013fd80b0.jpg): true **wire spool  -Material plastic -  Color oranges** -> predicted **wire spool /  rubber / oranges -  Pdfname  ZryFz3DBSiO8** (0.66)
- `results/poor_localization/example_1.jpg` (20260911_151239_d70c_primary_split_r.rf.10e0b7144bee60b66f2e2cf2f247cacf.jpg): true **book  / paper  / red** -> predicted **book  / paper  / red** (0.76)
- `results/poor_localization/example_2.jpg` (20260921_145959_0b29_primary.rf.f4a6b4977846bf1cfc66217ab824ea15.jpg): true **wire spool /  rubber / oranges -  Pdfname  ZryFz3DBSiO8** -> predicted **wire spool /  rubber / oranges -  Pdfname  ZryFz3DBSiO8** (0.61)
- `results/missed_low_confidence/example_1.jpg` (20260925_164514_3c4e_primary.rf.ce0fac426359af215658f6ff845e46a6.jpg): true **book  / paper  / red** -> predicted **book  / paper  / red** (0.26)
- `results/missed_low_confidence/example_2.jpg` (20260925_180711_24a2_primary.rf.c00918534c31121fa41382affbc85f6e.jpg): true **plastic container lid / plastic / red -  Pdfname  ZC63DBkv6oJY** -> predicted **plastic container lid / plastic / red -  Pdfname  ZC63DBkv6oJY** (0.23)
- `results/false_positive_background/example_1.jpg` (20260911_150357_d428_primary.rf.4d62c5b9f157505cb18a197355efed01.jpg): true **-** -> predicted **screw packaging box / cardboard** (0.69)
- `results/false_positive_background/example_2.jpg` (20260925_142407_c4f5_primary.rf.e1e78b784962694b8abf3c3de9819deb.jpg): true **-** -> predicted **desktop tape dispenser / plastic  ** (0.78)
- `results/false_positive_duplicate/example_1.jpg` (20260914_164432_88dd_primary.rf.587668f493d26ea22734534cc67ad218.jpg): true **-** -> predicted **retractable gel pen / rubber** (0.38)
- `results/false_positive_duplicate/example_2.jpg` (20260930_154636_b8b3_primary.rf.196d9a2b489b29978cb7899201aa3b7d.jpg): true **-** -> predicted **- extruded aluminum profile -  Material metal** (0.31)
