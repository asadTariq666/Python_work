# writing to file
file1 = open('output3.txt', 'w')
file1.close()
count = 0
Z = 0
X= 0
Y =0
flag = 0  # Ignore everything before this flag is set true
# variables ( just make changes by keeping up the format)
var1 ="[0.000000,0.000000,1.000000,0.000000]"
var2 = "[0,0,0,0]"
var3 = "[9E9,9E9,9E9,9E9,9E9,9E9]]"
var4 = "printSpeed"
var5 = "z0"
var6 = "bendedTool"
var7 = "\WObj:=wobj0;"

# variables at the start of file
s0_var = "MODULE MainModule"
s1_var = "VAR num delayOverhang     := 0;"
s2_var = "VAR num delayClosedStart  := 0.1;"
s3_var = "VAR num delayClosedStop   := 0.1;"
s4_var = "VAR num delayOpenStart    := 0.2;"
s5_var = "VAR num delayOpenStop     := 0.2;"
s6_var = "VAR speeddata printSpeed := [15.0, 500.0, 0.0, 0.0];"
s7_var = "VAR speeddata moveSpeed := [800.0, 500.0, 0.0, 0.0];"
s8_var = "VAR num wTime_20 := 2.0;"
s9_var = "PROC main()"
s10_var = "interpassTemp := 200.0;"
s11_var = "emissivity := 0.99;"
flag3=1

start_weld = 'waitInterpass True, wTime;\nstartContinuousWeld \delay:=delayClosedStart \weldingProgram:=1 \weldId:="0000-0000-0000-0002";'
stop_weld = 'stopContinuousWeld \delay:=delayClosedStop;'

end_var = '\nENDPROC\n\nENDMODULE'
f = open("input3.txt", "r")
w = open("output3.txt", "w")
# Adding variables to the start of file
start_text = s0_var+"\n"+s1_var+'\n'+s2_var+'\n'+s3_var+'\n'+s4_var+'\n'+s5_var+'\n'+s6_var+'\n'+s7_var+'\n'+s8_var+'\n\n'+s9_var+'\n'+s10_var+'\n'+s11_var+'\n\n'
w.write(start_text)
index =0
flag2 = 0 # For the first start weld.
flag3= 0 # for first Move J
# Reading Lines to extract codes from required lines and write them to the output file
Lines = f.readlines()
for line in Lines:
    if line.__contains__('End of Gcode'): # Writing ending text + stop weld + ending the program
        w.write(stop_weld+'\n')
        w.write(end_var+'\n')
        break
    if line.__contains__('LAYER_COUNT'):
        layers_count = line.replace(";LAYER_COUNT:","")
        layers_count = int(layers_count)


    if line.__contains__('LAYER:0'):  #  ignore everything before Layer: 0 
        flag =1
    if (flag):
        if line.__contains__('Z'):   # Getting values of Z
            splitted_line = line.split()
            if line.__contains__('F'): # if the line contains F code, the Z is at 4th index else at 3rd index
            #print(line,count)
                Z = splitted_line[4].replace("Z", "")
        if (line.__contains__(";TYPE:WALL-OUTER") and flag2 ==0):
            w.write(start_weld+'\n')
            flag2 =1
        # 1st Move J
        if((line.__contains__('G0')) and line.__contains__('X') and line.__contains__('Y') and flag3 ==0):
            splitted_line = line.split()
            for i in range(len(splitted_line)):
                if(splitted_line[i].__contains__('G0')):
                    move = splitted_line[i].replace("G0", "MoveJ")
                if(splitted_line[i].__contains__('X')):
                    X = splitted_line[i].replace("X", "")
                if(splitted_line[i].__contains__('Y')):
                    Y = splitted_line[i].replace("Y", "")
                if(splitted_line[i].__contains__('Z')):
                    Z = splitted_line[i].replace("Z", "") 
                flag3=1
            new_line = move+'[['+X+','+Y+','+Z+ "]," +var1+  ','  + var2  +','  + var3  +',' + var4  +',' + var5 +',' + var6 +',' + var7 + "\n"
            w.write(new_line)

        if((line.__contains__('G1 ')) and line.__contains__('X') and line.__contains__('Y')):
            splitted_line = line.split()
            for i in range(len(splitted_line)):
                if(splitted_line[i].__contains__('G1')):
                    move = splitted_line[i].replace("G1", "MoveL")
                if(splitted_line[i].__contains__('X')):
                    X = splitted_line[i].replace("X", "")
                if(splitted_line[i].__contains__('Y')):
                    Y = splitted_line[i].replace("Y", "")
                if(splitted_line[i].__contains__('Z')):
                    Z = splitted_line[i].replace("Z", "")
                
            new_line = move+'[['+X+','+Y+','+Z+ "]," +var1+  ','  + var2  +','  + var3  +',' + var4  +',' + var5 +',' + var6 +',' + var7 + "\n"
            w.write(new_line)
    index = index+1
f.close()
w.close()

