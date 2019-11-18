import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import datetime
import pandas as pd
import os

# sigmoid -
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    # init - Creates 3 weights made from random values between x, y, and the
    # nodes.
    # Purpose of weights:
    # x = obs
    # y = The feedfoward output
    # layer1_nodes =
    # layer2_nodes =
    # input:
    # weights1:
    # weights2:
    # weights3:
    # output:
    def __init__(self, x, y, layer1_nodes, layer2_nodes, reward):
        self.input = x
        self.weights1 = np.random.rand(len(x), layer1_nodes)
        self.weights2 = np.random.rand(layer1_nodes, layer2_nodes)
        self.weights3 = np.random.rand(layer2_nodes, len(y))
        self.output = np.zeros(len(y))
        self.reward = reward

    # feedforward - Creates 2 layers and an output from the weights.
    # Will determine which direction is chosen based on which value is greater.
    # Purpose of layers:
    # layer1:
    # layer2:
    # output:
    def feedforward(self, x):
        self.input = x
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer2, self.weights3))
        return (self.output)


class GA:
    # Initializes based on certain parameters
    # WOuld like to add Inner and Outer layer NN parameters
    def __init__(self, Size=20, Generation=10, Top=10, MutationRate=40, Dir="TestDefault", Case=None,
                 NameEnv="TimePilot-ram-v0", steps=5000):
        self.size = self.zeroCheck(Size)
        self.generations = self.zeroCheck(Generation)
        self.Top = self.doubleCheck(Top)
        self.mutation_rate = self.zeroCheck(MutationRate)
        # File Directory
        self.dir = Dir
        # Atari Environment
        self.decisions = self.decisions_env(NameEnv)
        self.env = gym.make(NameEnv)
        self.steps = steps

        if Case is None:
            self.generations = Size // 2
            self.Top = Size // 2

        self.data_structures_init_()
        self.mainCode()

    # Function to initialize
    def decisions_env(self, name):
        if name == "TimePilot-ram-v0":
            return 10
        elif name == "Breakout-ram-v0":
            return 4
        return 10  # just rerturn TimePilot command as default

    # The rest of the initialized variables that we will be using
    def data_structures_init_(self):
        if (self.dir == "TestDefault"):
            self.dir += str(datetime.datetime.today())
        if not (os.path.exists(self.dir)):
            os.mkdir(self.dir)
        else:
            print(self.dir + " already exists as a directory")
        self.net = []

        self.val = 0
        self.aver = 0

        self.highlist = []
        self.topper = []
        self.topAver = 0

        # Plotting Lists
        self.vallist = []
        self.averlist = []
        self.averListT = []
        self.RList = []

        # Output text
        self.StringText = []

    # Some quick error checking for bounds
    def zeroCheck(self, Z):
        if 0 >= Z:
            return 1
        return Z

    def checkTop(self, T):
        if T > self.size - 1:
            return self.size - 1
        return T

    def doubleCheck(self, D):
        if (D != self.zeroCheck(D)):
            return self.zeroCheck(D)
        if (D != self.checkTop(D)):
            return self.checkTop(D)
        return D

    def game_runner(self, NN):
        self.env.reset()
        total = 0
        action = self.env.action_space.sample()
        obs, reward, done, info = self.env.step(action)
        #self.env.render()
        total += reward
        for i in range(self.steps):
            actions = NN.feedforward(obs)
            action1 = 0
            # 10 - Time Pilot
            # 4 - Breakout
            for j in range(self.decisions):
                if actions[j] == max(actions):
                    action1 = j
                obs, reward, done, info = self.env.step(action1)
                #self.env.render()
                total += reward
            if done:
                break
        self.env.close()
        NN.reward = total

    # Function to create a specified number of neural networks.
    # ACTION_MEANING = {0: "NOOP",1: "FIRE",2: "UP",3: "RIGHT",4: "LEFT",5: "DOWN",
    # 6: "UPRIGHT",7: "UPLEFT",8: "DOWNRIGHT",9: "DOWNLEFT",10: "UPFIRE",11: "RIGHTFIRE",
    # 12: "LEFTFIRE",13: "DOWNFIRE",14: "UPRIGHTFIRE",15: "UPLEFTFIRE",16: "DOWNRIGHTFIRE",
    # 17: "DOWNLEFTFIRE",
    # }

    def neural_generation2(self, num, networks=None):
        group = []
        actions = []
        if networks is None:
            action = self.env.action_space.sample()
            # print("Action sample: ", action)
            obs, reward, done, info = self.env.step(action)
            n1 = NeuralNetwork(obs, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 6, 9, 0)
            group.append(n1)
            actions = n1.feedforward(obs)
            # print(actions)

            for j in range(num - 1):
                action1 = 0
                # 4 - Breakout
                # 10 - Time Pilot
                for i in range(self.decisions):
                    if actions[i] == max(actions):
                        action1 = i

                obs, reward, done, info = self.env.step(action1)
                n1 = NeuralNetwork(obs, actions, 6, 9, 0)
                group.append(n1)
                actions = n1.feedforward(obs)
        elif networks is not None:
            for k in range(num):
                action1 = 0
                actions = networks[k].feedforward(networks[k].input)
                # 4 - Breakout
                # 10 - Time Pilot:
                for j in range(self.decisions):
                    # for i in range(4):
                    if actions[j] == max(actions):
                        action1 = j
                obs, reward, done, info = self.env.step(action1)
                n1 = NeuralNetwork(obs, actions, 6, 9, 0)
                group.append(n1)
        return (group)

    def neural_fit(self, NN):
        # top = size // 2
        Best = []
        topper = len(NN) // 2
        if len(NN) % 2 != 0:
            topper += 1  # add 1 to topper
        for k in range(len(NN)):
            Best.append(NN[k].reward)
        # Gets top half, based on a sorted list of highest to lowest rewards
        Best_Ind = np.argsort(Best)[
                   -int(topper):]  # Gets the highest rewards returnning an array of Top contenders indices
        Best_Val = [Best[i] for i in Best_Ind]

        print("Best Fitness Value Recorded: ", max(Best_Val))
        self.StringText.append("Best Fitness Value Recorded: " + str(max(Best_Val)) + "\n")

        top = max(Best_Val)

        Used = []
        for i in range(len(NN)):
            if i in Best_Ind:
                Used.append(NN[i])
                if NN[i].reward == top:
                    n = NN[i]
        Used = sorted(Used, key=lambda NeuralNetwork: NeuralNetwork.reward)
        return (Used, top, n)

    # Will mutate the list of actions taken in order to increase number of iterations.
    # if the chance is less than rate, there is a second (binary) chance you still might
    # get a mutation
    # 0 - 100 to 50 - 50 where the change is * 0.05
    def mutation(self, weight, rate):
        new = 0
        chance = random.randint(0, 100)
        second = random.randint(0, 1)
        if chance < rate:
            # print(weight)
            if second == 0:
                new = float(weight) + (float(weight) * 0.05)
            else:
                new = float(weight) - (float(weight) * 0.05)
        else:
            new = weight
        return new

    # Will cross the weights of 2 networks together. Works on one weight pair.
    def crossover(self, NN1, NN2):
        cross1 = ''
        cross2 = ''
        weightA = str(NN1)

        weightB = str(NN2)

        for i in range(len(weightA) // 2):
            if (weightA[i] != "."):
                cross1 += weightA[i]

        for i in range(len(weightB) // 2):
            if (weightB[i + len(weightB) // 2] != "."):
                cross1 += weightB[i + len(weightB) // 2]

        for i in range(len(weightA) // 2):
            if (weightA[i + len(weightA) // 2] != "."):
                cross2 += weightA[i + len(weightA) // 2]

        for i in range(len(weightB) // 2):
            if (weightB[i] != "."):
                cross2 += weightB[i]

        cross11 = float(cross1[:1] + '.' + cross1[1:])
        cross22 = float(cross2[:1] + '.' + cross2[1:])
        return cross11, cross22

    def mainCode(self):
        mTopper = []
        self.env.reset()
        adder = self.neural_generation2(self.size)
        for l in range(len(adder)):
            self.net.append(adder[l])

        # Runs population for generations amount
        for z in range(self.generations):
            print("Run Number ", z + 1)
            self.StringText.append('\n')
            self.StringText.append("Run Number " + str(z + 1) + "\n")
            # net holds all the finished NN after 5000 steps (or once a goal was reached)
            TList = []
            for l in range(len(self.net)):
                self.game_runner(self.net[l])
                TList.append(self.net[l].reward)
                print("Network: ", l + 1)
            self.RList.append(TList)
            self.aver = 0
            # Fitness function of net, cutting back to only half of the fittest population
            self.net, val, ref = self.neural_fit(self.net)

            # The role of topper is to hold the unchanged top contender of the previous generation
            # If topper doesn't exist. creates it
            if (len(self.topper) == 0):
                self.topper = self.net
            # After the first generation, toppper will always be declared and holds the top contenders for each generation
            if (len(self.topper) != 0):
                for l in range(len(self.net)):
                    if (self.net[l].reward > self.topper[l].reward):
                        self.topper[l] = self.net[l]

            #                   #
            # Computer averages #
            #                   #
            rT = []
            for i in self.topper:
                self.topAver += i.reward
                rT.append(i.reward)
            self.topAver = self.topAver // len(self.topper)
            self.topper = sorted(self.topper, key=lambda NeuralNetwork: NeuralNetwork.reward)
            mTopper.append(max(rT))

            for i in self.net:
                self.aver += i.reward
            self.aver = self.aver // len(self.net)

            # Display Averages on console
            print("Average for the more fit half:", self.aver)
            print("Average for Topper List: ", self.topAver)

            self.StringText.append("Average for the more fit half: " + str(self.aver) + "\n")
            self.StringText.append('Average for Topper List:' + str(self.topAver) + "\n")
            self.StringText.append('\n')

            # FOR PLOTTING each Generation
            self.averlist.append(self.aver)
            self.averListT.append(self.topAver)
            # net2, val2, ref2 = neural_fit(net2)
            self.vallist.append(val)

            # choose new children population based on net or topper with highest average
            if (len(self.topper) != 0 and self.topAver <= self.aver):
                new_half = self.neural_generation2(self.size // 2, self.net)
            elif (len(self.topper) != 0 and self.topAver > self.aver):
                new_half = self.neural_generation2(self.size // 2, self.topper)
                # print("   Topper Stayed")

            # combine children with parents
            for i in new_half:
                self.net.append(i)

            waiter1 = []
            waiter2 = []
            waiter3 = []
            for i in range(len(self.net)):
                # for length of weight1
                for j in range(len(self.net[i].weights1)):
                    # for length of array in weight1
                    for n in range(len(self.net[i].weights1[j])):
                        waiter1.append(self.net[i].weights1[j][n])

                for j in range(len(self.net[i].weights2)):
                    for n in range(len(self.net[i].weights2[j])):
                        waiter2.append(self.net[i].weights2[j][n])

                for j in range(len(self.net[i].weights3)):
                    for n in range(len(self.net[i].weights3[j])):
                        waiter3.append(self.net[i].weights3[j][n])

            # Divides population in half, betweenthe old and new, crosses over and generates a whole new population
            for i in range(len(waiter1) // 2):
                clone1, clone2 = self.crossover(waiter1[i], waiter1[i + (len(waiter1) // 2)])
                waiter1[i] = clone1
                waiter1[i + (len(waiter1) // 2)] = clone2

            for i in range(len(waiter2) // 2):
                clone3, clone4 = self.crossover(waiter2[i], waiter2[i + (len(waiter2) // 2)])
                waiter2[i] = clone3
                waiter2[i + (len(waiter2) // 2)] = clone4

            for i in range(len(waiter3) // 2):
                clone5, clone6 = self.crossover(waiter3[i], waiter3[i + (len(waiter3) // 2)])
                waiter3[i] = clone5
                waiter3[i + (len(waiter3) // 2)] = clone6

            for i in range(len(waiter1)):
                waiter1[i] = self.mutation(waiter1[i], self.mutation_rate)
            for i in range(len(waiter2)):
                waiter2[i] = self.mutation(waiter2[i], self.mutation_rate)
            for i in range(len(waiter3)):
                waiter3[i] = self.mutation(waiter3[i], self.mutation_rate)

            x = 0
            for i in range(len(self.net)):
                for j in range(len(self.net[i].weights1)):
                    for n in range(len(self.net[i].weights1[j])):
                        self.net[i].weights1[j][n] = waiter1[x]
                        x += 1

            xx = 0
            for i in range(len(self.net)):
                for j in range(len(self.net[i].weights2)):
                    for n in range(len(self.net[i].weights2[j])):
                        self.net[i].weights2[j][n] = waiter2[xx]
                        xx += 1

            xxx = 0
            for i in range(len(self.net)):
                for j in range(len(self.net[i].weights3)):
                    for n in range(len(self.net[i].weights3[j])):
                        self.net[i].weights3[j][n] = waiter3[xxx]
                        xxx += 1
        self.env.close()

        # # # # # # # # # # # #
        # GRAPHING & CSV DATA #
        # # # # # # # # # # # #

        # Strings for naming later
        strG = str(self.size) + "_Gen_" + str(self.generations) + "_Top_" + str(self.Top) + "_MuR_" + str(
            self.mutation_rate)
        strD = "_T_" + str(datetime.datetime.today())

        # Graphing Related
        tempRList = []
        for j in self.RList[0]:
            dList = []
            dList.append(j)
            tempRList.append(dList)
        for i in range(1, self.generations):
            k = 0
            for j in self.RList[i]:
                tempRList[k].append(j)
                k += 1
        self.RList = tempRList

        self.StringText.append("\n")

        print("Highest Value Recorded: ", max(self.vallist))
        self.StringText.append("Highest Value Recorded " + str(max(self.vallist)) + "\n")
        high_aver = 0
        for i in self.vallist:
            high_aver += i
        high_aver = high_aver // len(self.vallist)
        check = 0
        check = self.vallist[len(self.vallist) - 1] - self.vallist[0]
        print("\nAverage of highest values in each generation:", high_aver)
        print("Difference between highest values of start and end:", check, "\n")
        self.StringText.append("Average of highest values in each generation:" + str(high_aver) + "\n")
        self.StringText.append("Difference between highest values of start and end: " + str(check) + "\n")
        fig = plt.figure()
        ax = plt.subplot()
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.plot(self.vallist, 'ro', label="Best Gen Value")
        plt.plot(self.averListT, c='g', label="Avg Best")
        plt.plot(self.averlist, c='y', label="Avg Gen")
        ax.set_ylabel('Score')
        ax.set_title('Average and Highest Values')
        plt.legend()
        plt.show()

        cFig = self.dir + "/Gen_Plot_"
        endFig = ".png"
        realFigName = cFig + strG + strD + endFig
        fig.savefig(realFigName)

        fig = plt.figure()
        ax = plt.subplot()
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        DataR = pd.DataFrame()
        DataR['BestV'] = self.vallist
        DataR['AvgB'] = self.averListT
        DataR['AvgG'] = self.averlist

        counter = 1
        r = []
        for l in mTopper:
            r.append(l)
        plt.plot(r, c=np.random.rand(3, ))
        ax.set_ylabel('Highest Topper Score')
        ax.set_title('Overall Performance ' + str(self.size))
        name_R = "NN_" + str(counter)
        counter += 1
        DataR[name_R] = l
        plt.legend()
        plt.show()

        cFig = self.dir + "/NN_Topper"
        realFigName = cFig + strG + strD + endFig
        fig.savefig(realFigName)


        print(DataR)

        cCsv = self.dir + "/R_data_" + strG + strD + ".csv"
        export_csv = DataR.to_csv(cCsv, index=None, header=True)

        cCsv = self.dir + "/" + self.dir + " Logs.txt"
        f = open(cCsv, "w")
        f.writelines(self.StringText)
        f.close()


def autoLabel(rects, I, ax):
    cI = 0
    for rect in rects:
        height = rect.get_height()
        information = "P: " + str(I[cI][0]) + "\nC: " + str(I[cI][1]) + "\nF: " + str(I[cI][2]) + "\nM: " + str(
            I[cI][3])
        ax.annotate('{}'.format(height), xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va="bottom")
        ax.annotate('{}'.format(information), xy=(rect.get_x() + rect.get_width() / 2, height / 2), xytext=(0, 3),
                    textcoords="offset points", ha='center', va="bottom")
        cI += 1


def createGraph(testCaseResults, Cases, t):
    fig, ax = plt.subplots()
    tests = []
    for i in range(len(testCaseResults)):
        tests.append("test" + str(i + 1))

    r = ax.bar(np.arange(len(testCaseResults)), testCaseResults)
    ax.set_ylabel('Highest Score')
    ax.set_title('Overall Performance for each test case ' + str(int(t)))
    ax.set_xticks(np.arange(len(testCaseResults)))
    ax.set_xticklabels(tests)
    plt.xticks(np.arange(int(len(testCaseResults))))
    autoLabel(r, Cases, ax)
    fig.tight_layout()

    plt.show()
    if not (os.path.exists("TestOverall")):
        os.mkdir("TestOverall")
    else:
        print("TestOverall" + " already exists as a directory")

    dFig = "TestOverall/NN_Plot_" + str(t) + "_" + str(datetime.datetime.today()) + "_Overall.png"
    fig.savefig(dFig)


def createBarGraphforTest(TestCases, t):
    TestGA = []
    cT = 0
    # Basing it off of size, allows me to mix and match different scenarios, more easily
    for i in TestCases:
        if (len(i) - 1 < 5):  # 5 is the indices of the choice
            strN = i[4] + str(cT) + "_P_" + str(i[0]) + "_GH_TH_M_" + str(i[3])
            TestGA.append(max(GA(i[0], i[1], i[2], i[3], strN).vallist))
            i[1] = i[0] // 2  # GH
            i[2] = i[0] // 2  # TH
        else:  # means that the choice parameter was declared
            if i[5] == None:
                strN = i[4] + str(cT) + "_P_" + str(i[0]) + "_GH_TH_M_" + str(i[3])
                TestGA.append(max(GA(i[0], i[1], i[2], i[3], strN).vallist))
                i[1] = i[0] // 2  # GH
                i[2] = i[0] // 2  # TH
            elif i[5] == 1:
                strN = i[4] + str(cT) + "_P_" + str(i[0]) + "_G_" + str(i[1]) + "_T_" + str(i[2]) + "_M_" + str(i[3])
                TestGA.append(max(GA(i[0], i[1], i[2], i[3], strN, i[5]).vallist))
            else:
                strN = "TestDefault_P20_GH_TH_M40"
                TestGA.append(max(GA().vallist))
        cT += 1
    createGraph(TestGA, TestCases, t)


# File naming conventions
# first letter is the Gym it is in (in case of reptition continue the number)
# "test"
# letter representing which case it is in
# P = population, G = generation     H = Half of P
# T = fitess pop, M = Mutation Rate

################# # Population Size, Generation Length, Size of Fittest, Name of Test
#   Constants   # # We have 0 in GL and Fittest because in the algorithm it automatically disregards those
#################
# TimePilot doesn't need a variable as its the default environment
Breakout = "Breakout-ram-v0"

################# # Population Size, Generation Length, Size of Fittest, Name of Test
#     DEBUG     # # We have 0 in GL and Fittest because in the algorithm it automatically disregards those
#################

print("Base Initation Called...")

#####################
#                   #
#     TimePilot     #
#                   #
#####################

####################
# Debug Test Case  # # We have 0 in GL and Fittest because in the algorithm it automatically disregards those
#################### # unless you add a third parameter of 1 (or anything) after the name of the Test (seen in Test Case 2)

# GA( 25, 30, 10, 25, "B0TestA", 1, Breakout, 10000)
# GA() #TimePilot
# 5 Tests per case

TestGADB = []
print("TESTING")
TestGADB.append([4, 0, 0, 40, "DBTestA"])  # SIZE OF FIVE
TestGADB.append([8, 4, 4, 40, "DBTestA", 1])
#createBarGraphforTest(TestGADB, 7)

print("TIMEPILOT")

#################
#  TEST CASE 1  # # We have 0 in GL and Fittest because in the algorithm it automatically disregards those
################# # unless you add a third parameter of 1 (or anything) after the name of the Test (seen in Test Case 2)
TestGAA = []
print("TEST A Set 50/50")
TestGAA.append([16, 0, 0, 40, "ATestA"])  # SIZE OF FIVE
#TestGAA.append([32, 0, 0, 40, "ATestA"])
#TestGAA.append([64, 0, 0, 40, "ATestA"])
#TestGAA.append([128, 0, 0, 40, "ATestA"])
#TestGAA.append([84, 0, 0, 40, "ATestA"])

createBarGraphforTest(TestGAA, 1)

#################
#  TEST CASE 2  # # If you don't add the 1 at the end, the code will disregard your GL and Fitness values
#################

TestGAA2 = []
print("TEST B Set Diverse\n")
TestGAA2.append([30, 90, 18, 30, "ATestB", 1])
TestGAA2.append([12, 37, 4, 62, "ATestB", 1])
TestGAA2.append([50, 23, 10, 40, "ATestB", 1])
TestGAA2.append([100, 5, 25, 70, "ATestB", 1])
TestGAA2.append([5, 41, 3, 22, "ATestB", 1])

# createBarGraphforTest(TestGAA2, 2)

#################
#  TEST CASE 3  # # Mixed Bag
#################

TestGAA3 = []
print("TEST C Set Mixed")
TestGAA3.append([6, 64, 20, 45, "ATestC", 1])  # Large population, longer than half generation count, exclusiveness, moderate mutation
#TestGAA3.append([80, 0, 0, 45, "ATestC"])  # comparison with TestC1
#TestGAA3.append([18, 0, 0, 32, "ATestC"])  # low population, with below average mutation
#TestGAA3.append([16, 42, 10, 15, "ATestC", 1])  # low population with large generation count, but low mutation
#TestGAA3.append([8, 5, 8, 72, "ATestC", 1])  #

createBarGraphforTest(TestGAA3, 3)

#################
#  TEST CASE 4  # # Random Generates only 3 cases
#################

TestGAA4 = []
print("TEST D Random")
for i in range(3):
    C = random.randint(0, 1)
    A = random.randint(20, 150)  # Population
    M = random.randint(1, 100)  # Mutation
    if (C == 0):
        TestGAA4.append([A, 0, 0, M, "ATestD"])
    else:
        # Generations
        if A >= 100:
            Gen = random.randint(5, 85)
        else:
            Gen = random.randint(20, 50)
        # Fitness
        F = random.randint(int(A / 2), (A + int(A * (M / 100))))
        TestGAA4.append([A, Gen, F, M, "ATestD", 1])

# createBarGraphforTest(TestGAA4, 4)

######################
#                    #
#      Breakout      #
#                    #
######################
print("BREAKOUT")
#################
#  TEST CASE 1  # # We have 0 in GL and Fittest because in the algorithm it automatically disregards those
################# # unless you add a third parameter of 1 (or anything) after the name of the Test (seen in Test Case 2)
TestGAB = []
print("TEST A Set 50/50")
TestGAB.append([16, 0, 0, 40, "BTestA", None, Breakout, 100000])  # SIZE OF FIVE
TestGAB.append([32, 0, 0, 40, "BTestA", None, Breakout, 100000])
TestGAB.append([64, 0, 0, 40, "BTestA", None, Breakout, 100000])
TestGAB.append([128, 0, 0, 40, "BTestA", None, Breakout, 100000])
TestGAB.append([84, 0, 0, 40, "BTestA", None, Breakout, 100000])

#createBarGraphforTest(TestGAB, 6)

#################
#  TEST CASE 2  # # If you don't add the 1 at the end, the code will disregard your GL and Fitness values
#################

TestGAB2 = []
print("TEST B Set Diverse\n")
TestGAB2.append([30, 90, 18, 30, "BTestB", 1, Breakout, 100000])
TestGAB2.append([12, 37, 4, 62, "BTestB", 1, Breakout, 100000])
TestGAB2.append([50, 23, 10, 40, "BTestB", 1, Breakout, 100000])
TestGAB2.append([100, 5, 25, 70, "BTestB", 1, Breakout, 100000])
TestGAB2.append([5, 41, 3, 22, "BTestB", 1, Breakout, 100000])

#createBarGraphforTest(TestGAB2, 7)

#################
#  TEST CASE 3  # #Testing out a mix bag of different chosen values
#################

TestGAB3 = []
TestGAB3.append([26, 30, 10, 55, "BTestC", 1, Breakout, 100000])
TestGAB3.append([26, 0, 0, 55, "BTestC", None, Breakout, 100000])
TestGAB3.append([50, 30, 10, 25, "BTestC", 1, Breakout, 100000])
TestGAB3.append([50, 30, 10, 25, "BTestC", None, Breakout, 100000])
TestGAB3.append([128, 23, 64, 75, "BTestC", 1, Breakout, 100000])
TestGAB3.append([128, 23, 64, 5, "BTestC", 1, Breakout, 100000])
#createBarGraphforTest(TestGAB3, 8)

#################
#  TEST CASE 4  # # Random Generates only 3 cases
#################

TestGAB4 = []
print("TEST D Random")
for i in range(3):
    C = random.randint(0, 1)
    A = random.randint(20, 128)  # Population
    M = random.randint(1, 100)  # Mutation
    if (C == 0):
        TestGAB4.append([A, 0, 0, M, "BTestD", None, Breakout, 100000])
    else:
        # Generations
        Gen = random.randint(25, 130)
        # Fitness size
        F = random.randint(int(A / 2), (A + int(A * (M / 100))))
        TestGAB4.append([A, Gen, F, M, "BTestD", 1, Breakout, 100000])

#createBarGraphforTest(TestGAB4, 9)
