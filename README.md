# GA_and_RL
This is my test bank for different Genetic Algorithm and Reinforcement Learning implementations.
/bin holds:<br>
GA.py<br>
RL.py<br>
es.py<br>


To run make sure you have all the relevant dependencies installed:<br>
pip3 install gym<br>
... gym[atari]<br>
... atari-py<br>
... numpy<br>
... panda<br>
... cma<br>

All you need to do is run GA.py for the Genetic Algorithm and it will output the files to its root. The formatting in the rar file (if you run the code the generated code creates a different structure depicted at the bottom of the readme) is as such: 

This would be the example of a directory with 2 different environments with 2 sets of cases each with 2 cases

Please note that Topper is not in the zip file, but it will exist if you run GA.py, it was added after the data was generated
Some keys to remember
P: population G: Generation M: Mutation <br>
H: Half of P
>Root | GA.py
>Time Pilot
>>ATestA0_P_100_GH_TH_M_40   | Directory Set 1 Case 1
>>> ATestA0_P_100_GH_TH_M_40.txt | Directory set 1<br>
>>> Gen_Plot_100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> NN_Plot100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> R_data_100_Gen_50_Top_50_MuR_40_T_time.csv<br>
>>> Topper_100_Gen_50_Top_50_MuR_40_T_time.png<br>
>
>>ATestA1_P_200_GH_TH_M_40 | Directory  Set 1 Case 2
>>> ATestA1_P_200_GH_TH_M_40.txt | Directory set 2<br>
>>> Gen_Plot_200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> NN_Plot200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> R_data_200_Gen_100_Top_100_MuR_40_T_time.csv
>
>>ATestB0_P_100_GH_TH_M_40   | Directory Set 2 Case 1
>>> ATestB0_P_100_GH_TH_M_40.txt | Directory set 1<br>
>>> Gen_Plot_100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> NN_Plot100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> R_data_100_Gen_50_Top_50_MuR_40_T_time.csv<br>
>>> Topper_100_Gen_50_Top_50_MuR_40_T_time.png<br>
>
>>ATestB1_P_200_GH_TH_M_40 | Directory  Set 2 Case 2
>>> ATestB1_P_200_GH_TH_M_40.txt | Directory set 2<br>
>>> Gen_Plot_200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> NN_Plot200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> R_data_200_Gen_100_Top_100_MuR_40_T_time.csv
>Breakout
>>BTestA0_P_100_GH_TH_M_40   | Directory Set 1 Case 1
>>> BTestA0_P_100_GH_TH_M_40.txt | Directory set 1<br>
>>> Gen_Plot_100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> NN_Plot100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> R_data_100_Gen_50_Top_50_MuR_40_T_time.csv<br>
>>> Topper_100_Gen_50_Top_50_MuR_40_T_time.png<br>
>
>>BTestA1_P_200_GH_TH_M_40 | Directory  Set 1 Case 2
>>> BTestA1_P_200_GH_TH_M_40.txt | Directory set 2<br>
>>> Gen_Plot_200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> NN_Plot200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> R_data_200_Gen_100_Top_100_MuR_40_T_time.csv
>
>>BTestB0_P_100_GH_TH_M_40   | Directory Set 2 Case 1
>>> BTestB0_P_100_GH_TH_M_40.txt | Directory set 1<br>
>>> Gen_Plot_100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> NN_Plot100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> R_data_100_Gen_50_Top_50_MuR_40_T_time.csv<br>
>>> Topper_100_Gen_50_Top_50_MuR_40_T_time.png<br>
>
>>BTestB1_P_200_GH_TH_M_40 | Directory  Set 2 Case 2
>>> ATestB1_P_200_GH_TH_M_40.txt | Directory set 2<br>
>>> Gen_Plot_200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> NN_Plot200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> R_data_200_Gen_100_Top_100_MuR_40_T_time.csv

RL.py has the same output on the rar as well as in the code, and wish to model GA similarly in the future:

Note, a mistake we made was that the label uses F instead of G for generation

This would be the example of a directory with 2 different environments with 2 sets of cases each with 2 cases
for every case it iterates, it increases the count of the population by the count
while for every set the Generation is increased by its counter

>Root | RL.py
>Breakout
>> _RL_A0
>>> Grid_Plot__HL1_64_HL2_64_P_50_F_500.png<br>
>>> Logs_data__HL1_64_HL2_64_P_50_F_500.txt<br>
>>> R_data__HL1_64_HL2_64_P_50_F_500.csv
>
>>_RL_A1
>>> Grid_Plot__HL1_64_HL2_64_P_100_F_500.png<br>
>>> Logs_data__HL1_64_HL2_64_P_100_F_500.txt<br>
>>> R_data__HL1_64_HL2_64_P_100_F_500.csv
>
>> _RL_B0
>>> Grid_Plot__HL1_64_HL2_64_P_50_F_1000.png<br>
>>> Logs_data__HL1_64_HL2_64_P_50_F_1000.txt<br>
>>> R_data__HL1_64_HL2_64_P_50_F_500.csv
>
>> _RL_A1
>>> Grid_Plot__HL1_64_HL2_64_P_100_F_1000.png<br>
>> Logs_data__HL1_64_HL2_64_P_100_F_1000.txt<br>
>>> R_data__HL1_64_HL2_64_P_100_F_1000.csv
>
>TimePilot
>>> Grid_Plot__HL1_64_HL2_64_P_50_F_500.png<br>
>>> Logs_data__HL1_64_HL2_64_P_50_F_500.txt<br>
>>> R_data__HL1_64_HL2_64_P_50_F_500.csv
>
>> _RL_A1
>>> Grid_Plot__HL1_64_HL2_64_P_100_F_500.png<br>
>>> Logs_data__HL1_64_HL2_64_P_100_F_500.txt<br>
>>> R_data__HL1_64_HL2_64_P_100_F_500.csv
>
>> _RL_B0
>>> Grid_Plot__HL1_64_HL2_64_P_50_F_1000.png<br>
>>> Logs_data__HL1_64_HL2_64_P_50_F_1000.txt<br>
>>> R_data__HL1_64_HL2_64_P_50_F_500.csv
>
>> _RL_A1
>>> Grid_Plot__HL1_64_HL2_64_P_100_F_1000.png<br>
>>> Logs_data__HL1_64_HL2_64_P_100_F_1000.txt<br>
>>> R_data__HL1_64_HL2_64_P_100_F_1000.csv


Current GA outputs: <We will soon have them output to their own folders, to not cluter root>
>Root | GA.py
>Time Pilot
>>ATestA0_P_100_GH_TH_M_40   | Directory Set 1 Case 1
>>> ATestA0_P_100_GH_TH_M_40.txt | Directory set 1<br>
>>> Gen_Plot_100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> NN_Topper100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> R_data_100_Gen_50_Top_50_MuR_40_T_time.csv<br>
>
>>ATestA1_P_200_GH_TH_M_40 | Directory  Set 1 Case 2
>>> ATestA1_P_200_GH_TH_M_40.txt | Directory set 2<br>
>>> Gen_Plot_200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> NN_Topper200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> R_data_200_Gen_100_Top_100_MuR_40_T_time.csv
>
>>ATestB0_P_100_GH_TH_M_40   | Directory Set 2 Case 1
>>> ATestB0_P_100_GH_TH_M_40.txt | Directory set 1<br>
>>> Gen_Plot_100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> NN_Topper100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> R_data_100_Gen_50_Top_50_MuR_40_T_time.csv<br>
>
>>ATestB1_P_200_GH_TH_M_40 | Directory  Set 2 Case 2
>>> ATestB1_P_200_GH_TH_M_40.txt | Directory set 2<br>
>>> Gen_Plot_200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> NN_Topper200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> R_data_200_Gen_100_Top_100_MuR_40_T_time.csv
>Breakout
>>BTestA0_P_100_GH_TH_M_40   | Directory Set 1 Case 1
>>> BTestA0_P_100_GH_TH_M_40.txt | Directory set 1<br>
>>> Gen_Plot_100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> NN_Topper100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> R_data_100_Gen_50_Top_50_MuR_40_T_time.csv<br>
>
>>BTestA1_P_200_GH_TH_M_40 | Directory  Set 1 Case 2
>>> BTestA1_P_200_GH_TH_M_40.txt | Directory set 2<br>
>>> Gen_Plot_200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> NN_Topper200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> R_data_200_Gen_100_Top_100_MuR_40_T_time.csv
>
>>BTestB0_P_100_GH_TH_M_40   | Directory Set 2 Case 1
>>> BTestB0_P_100_GH_TH_M_40.txt | Directory set 1<br>
>>> Gen_Plot_100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> NN_Topper100_Gen_50_Top_50_MuR_40_T_time.png<br>
>>> R_data_100_Gen_50_Top_50_MuR_40_T_time.csv<br>
>
>>BTestB1_P_200_GH_TH_M_40 | Directory  Set 2 Case 2
>>> ATestB1_P_200_GH_TH_M_40.txt | Directory set 2<br>
>>> Gen_Plot_200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> NN_Topper200_Gen_100_Top_100_MuR_40_T_time.png<br>
>>> R_data_200_Gen_100_Top_100_MuR_40_T_time.csv