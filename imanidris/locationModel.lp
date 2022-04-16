\* MedicalCenterLocationModel *\
Minimize
OBJ: __dummy
Subject To
_C1: x_0_Downtown_Toronto + x_1_Downtown_Toronto + x_2_Downtown_Toronto
 + x_3_Downtown_Toronto + x_4_Downtown_Toronto + x_5_Downtown_Toronto
 = 1237519
_C10: x_2_Downtown_Toronto + x_2_East_York + x_2_Etobicoke
 + x_2_Mississauga_East + x_2_North_Toronto + x_2_North_York
 + x_2_Scarborough_East - 5000000 y_2 <= 0
_C11: x_3_Downtown_Toronto + x_3_East_York + x_3_Etobicoke
 + x_3_Mississauga_East + x_3_North_Toronto + x_3_North_York
 + x_3_Scarborough_East - 5000000 y_3 <= 0
_C12: x_4_Downtown_Toronto + x_4_East_York + x_4_Etobicoke
 + x_4_Mississauga_East + x_4_North_Toronto + x_4_North_York
 + x_4_Scarborough_East - 5000000 y_4 <= 0
_C13: x_5_Downtown_Toronto + x_5_East_York + x_5_Etobicoke
 + x_5_Mississauga_East + x_5_North_Toronto + x_5_North_York
 + x_5_Scarborough_East - 5000000 y_5 <= 0
_C14: x_0_Downtown_Toronto + x_0_East_York + x_0_Etobicoke
 + x_0_Mississauga_East + x_0_North_Toronto + x_0_North_York
 + x_0_Scarborough_East <= 772500
_C15: x_1_Downtown_Toronto + x_1_East_York + x_1_Etobicoke
 + x_1_Mississauga_East + x_1_North_Toronto + x_1_North_York
 + x_1_Scarborough_East <= 637500
_C16: x_2_Downtown_Toronto + x_2_East_York + x_2_Etobicoke
 + x_2_Mississauga_East + x_2_North_Toronto + x_2_North_York
 + x_2_Scarborough_East <= 600000
_C17: x_3_Downtown_Toronto + x_3_East_York + x_3_Etobicoke
 + x_3_Mississauga_East + x_3_North_Toronto + x_3_North_York
 + x_3_Scarborough_East <= 630000
_C18: x_4_Downtown_Toronto + x_4_East_York + x_4_Etobicoke
 + x_4_Mississauga_East + x_4_North_Toronto + x_4_North_York
 + x_4_Scarborough_East <= 450000
_C19: x_5_Downtown_Toronto + x_5_East_York + x_5_Etobicoke
 + x_5_Mississauga_East + x_5_North_Toronto + x_5_North_York
 + x_5_Scarborough_East <= 450000
_C2: x_0_East_York + x_1_East_York + x_2_East_York + x_3_East_York
 + x_4_East_York + x_5_East_York = 857483
_C3: x_0_Etobicoke + x_1_Etobicoke + x_2_Etobicoke + x_3_Etobicoke
 + x_4_Etobicoke + x_5_Etobicoke = 1426537
_C4: x_0_Mississauga_East + x_1_Mississauga_East + x_2_Mississauga_East
 + x_3_Mississauga_East + x_4_Mississauga_East + x_5_Mississauga_East
 = 1267191
_C5: x_0_North_Toronto + x_1_North_Toronto + x_2_North_Toronto
 + x_3_North_Toronto + x_4_North_Toronto + x_5_North_Toronto = 1299074
_C6: x_0_North_York + x_1_North_York + x_2_North_York + x_3_North_York
 + x_4_North_York + x_5_North_York = 1585282
_C7: x_0_Scarborough_East + x_1_Scarborough_East + x_2_Scarborough_East
 + x_3_Scarborough_East + x_4_Scarborough_East + x_5_Scarborough_East
 = 1388656
_C8: x_0_Downtown_Toronto + x_0_East_York + x_0_Etobicoke
 + x_0_Mississauga_East + x_0_North_Toronto + x_0_North_York
 + x_0_Scarborough_East - 5000000 y_0 <= 0
_C9: x_1_Downtown_Toronto + x_1_East_York + x_1_Etobicoke
 + x_1_Mississauga_East + x_1_North_Toronto + x_1_North_York
 + x_1_Scarborough_East - 5000000 y_1 <= 0
Bounds
__dummy = 0
Binaries
y_0
y_1
y_2
y_3
y_4
y_5
End
