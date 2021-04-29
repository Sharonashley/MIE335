import pandas as pd
import numpy as np 
import scipy as sp
import queue as Q
from gurobipy import *
import itertools
import matplotlib.pyplot as plt 
import copy
import time
import collections
import os

def read_input(path):
    file = open(path,"r+") 
    
    # initialize Variables
    first_line = True
    num_x = None
    constraint_rhs = []
    objectives = []
    constraints = []

    for line in file:

        curr_line = line.strip() #.split()    
        if first_line: # check if first line
            num_x = int(curr_line)
            first_line = False
        elif (' ' not in curr_line): # check if RHS of constraints
            constraint_rhs.append(int(curr_line))
        elif line[0] == '-':
            curr_line = [int(i) for i in curr_line.split()]
            objectives.append(curr_line)
        else:
            curr_line = [int(i) for i in curr_line.split()]
            constraints.append(curr_line)

    return num_x, constraint_rhs, objectives, constraints

def export_NDPs(filename, NDPs):
    # remove duplicates
    NDPs = np.unique(np.array(NDPs), axis=0)

    # sort the array in descending lexmin order:
    ind = []
    for i in range(len(NDPs[0])-1,-1,-1):
        ind.append(-NDPs[:,i])
    NDPs = NDPs[np.lexsort(ind)]

    directory = os.getcwd() + "/"
    np.savetxt(directory + filename, NDPs, delimiter='\t', newline='\n')
    
def export_summary(filename, summary):
    directory = os.getcwd() + "/"
    np.savetxt(directory + filename, np.array(summary), delimiter='\t', newline='\n')


def brute_force(num_x, constraint_rhs, objectives, constraints):

    # start time
    start = time.time()
    
    def find_feasible_points(constraints, constraints_rhs, n):        
        # manually generate binary combinations
        combinations = []

        for i in range(1<<n):
            s = bin(i)[2:] # get binary representation
            s = s='0'*(n-len(s))+s # prepend missing 0's 
            s = list(int(i) for i in str(s)) # convert to list
            combinations.append(s) # append to combinations

        # initialize feasible list
        feas = []

        # test which combinations meet constraint
        for comb in combinations:
            is_feas = True
            for i in range(len(constraints)):
                # multiply combination with constraint and sum up
                mult_sum = np.sum(np.multiply(comb,constraints[i]))
                if mult_sum > constraints_rhs[i]:
                    is_feas = False #if it does not fall within one of the constraints, mark as infeasible
            if is_feas == True:
                feas.append(comb) #if it is feasible, add to list

        # enumerate feasible combinations
        feasible = {} #add to dictionary
        i = 1
        for f in feas:
            feasible[i] = f
            i += 1

        return feasible
    
    def find_obj_equivalent(feasible_points, objectives):
        obj = {}
        
        # for each value in the dictionary, and for each objective
        # calculate the Z value and save them to a list as the value in a dictionary
        for key, value in feasible_points.items():
            curr_obj = []
            for objective in objectives:
                curr_obj.append(float(np.sum(np.multiply(value,objective))))
            obj[key] = curr_obj

        return obj
    
    def find_NDF(obj_equivalent):
        dominated = []
        objective_points = list(obj_equiv.values())
        
        # iterate through each feasible Z value as current point to check if dominated
        for i in range(len(objective_points)):
            curr_point = objective_points[i]
            for j in range(len(objective_points)): 
                # iterate through each other feasible Z value to check if it dominates current point
                comparison_point = objective_points[j]
                if curr_point != comparison_point: 
                    dom = True # do not compare with itself
                    for k in range(len(comparison_point)):
                        if curr_point[k] < comparison_point[k]:
                            dom = False 
                    # if at least one of the current points Z value is less than the comparison point, 
                    # it is not dominated by this point
                    if dom == True:
                        dominated.append(curr_point) 
                        # if it is dominated, add to the list of dominated points and break the loop, 
                        # no need to check more points if we already know it is dominated
                        break 

        # use Numpy Set Subtraction to find NDF
        a = np.array(objective_points)
        b = np.array(dominated)
        a1_rows = a.view([('', a.dtype)] * a.shape[1])
        a2_rows = b.view([('', b.dtype)] * b.shape[1])
        NDF = np.setdiff1d(a1_rows, a2_rows).view(a.dtype).reshape(-1, a.shape[1])
        # NDF is defined as the set resulting from subtracting the set of dominated points by the set 
        # of feasible points
        
        return NDF

    feasible_ex = find_feasible_points(constraints,constraint_rhs,num_x)
    obj_equiv = find_obj_equivalent(feasible_ex,objectives)
    NDFs = find_NDF(obj_equiv)
    
    elapsed = (time.time() - start)
    summary = [elapsed, len(NDFs), 0]
    
    return NDFs, summary


def rectangular(num_x, constraints, constraint_rhs, objectives): 
    
    # start timer
    start = time.time()
    
    # ------------ Helper Methods ------------
    
    def lexmin_initial(n, b, o, c, flag):
        # initialize model 
        m = Model('Solve Lexmin')
        m.setParam("Threads", 1)
        m.Params.OutputFlag = 0
        m.Params.MIPGap = 1e-6
        
        # initialize binary variable
        variable = m.addVars(n, vtype=GRB.BINARY, name="variable")

        # set objective based on lexmin(z1,z2) or lexmin(z2,z1)
        if flag == 1:
            obj = LinExpr(quicksum([variable[i]*o[0][i] for i in range(n)]))
        elif flag == 2:
            obj = LinExpr(quicksum([variable[i]*o[1][i] for i in range(n)]))
        m.setObjective(obj, GRB.MINIMIZE)

        # add knapsack constraints
        for k in range(len(c)):
            constraint = m.addConstr(quicksum([c[k][i]*variable[i] for i in range(n)]) <= b[k])
        
        # update model and find optimal objective value
        m.update()
        m.optimize()
        opt_var = [np.round(variable[i].x) for i in range(n)] # prevent numerical issues which occur without rounding
        if flag == 1:
            optimal_z1 = int(sum(opt_var*(np.array(o[0]))))
        elif flag == 2:
            optimal_z1 = int(sum(opt_var*(np.array(o[1]))))

        # minimize other objective based on lexmin(z1,z2) or lexmin(z2,z1)
        if flag == 1:
            new_obj = LinExpr(quicksum([variable[i]*o[1][i] for i in range(n)]))
            m.addConstr(quicksum([variable[i]*o[0][i] for i in range(n)]) <= optimal_z1)
        elif flag == 2:
            new_obj = LinExpr(quicksum([variable[i]*o[0][i] for i in range(n)]))
            m.addConstr(quicksum([variable[i]*o[1][i] for i in range(n)]) <= optimal_z1)
        m.setObjective(new_obj, GRB.MINIMIZE)

        # update model and find optimal objective value
        m.update()
        m.optimize()
        opt_var = [np.round(variable[i].x) for i in range(n)] # prevent numerical issues which occur without rounding
        if flag == 1: 
            optimal_z2 = int(sum(opt_var*(np.array(o[1]))))
        elif flag == 2:
            optimal_z2 = int(sum(opt_var*(np.array(o[0]))))

        # return optimal [z*[0], z*[1]] and change ordering based on lexmin(z1,z2) or lexmin(z2,z1)
        if flag == 1:
            return [optimal_z1, optimal_z2]
        elif flag == 2:
            return [optimal_z2, optimal_z1]
    
    
    # ------------ Initializing Parameters and Data Structures ------------
    
    # intialize parameters and list to track found NDPs
    foundNDPs = []
    epsilon = 1
    
    
    # ------------ Find ZNW, and ZSE and initial Rectangle using helper method ------------
    
    # initial zNW
    zNW = lexmin_initial(num_x, constraint_rhs, objectives, constraints, 1)
    zNW = np.round(zNW)
    foundNDPs.append(zNW)

    # initial zSE
    zSE = lexmin_initial(num_x, constraint_rhs, objectives, constraints, 2)
    zSE = np.round(zSE)
    foundNDPs.append(zSE)
       
    #initialize the list of rectangles
    Rectangles = [[zNW, zSE]]
    
    # ------------ Initialize Model to be Used for Dynamically Updating Constraints ------------
    m = Model('Solve Lexmin')
    m.setParam("Threads", 1)
    m.Params.OutputFlag = 0 # suppress output
    m.Params.MIPGap = 1e-6 # updated tolerance limit
    m.modelSense = GRB.MINIMIZE 
    variable = m.addVars(num_x, vtype=GRB.BINARY, name="variable") # binary variables X_i

    # define objective function linear expressions
    z1_obj = LinExpr(quicksum([variable[i]*objectives[0][i] for i in range(num_x)]))
    z2_obj = LinExpr(quicksum([variable[i]*objectives[1][i] for i in range(num_x)]))
    
    # add general knapsack constraints 
    for k in range(len(constraints)):
        m.addConstr(quicksum([constraints[k][i]*variable[i] for i in range(num_x)]) <= constraint_rhs[k], "knapsackconstraint{0}".format(k))
    
    # ------------ Explore Rectangles ------------
    
    # initialize count of number of rectangles explored
    count = 0
    
    while Rectangles:
        
        count += 1 # increment cound
        R = Rectangles[0] # arbitrarily pick first rectangle to be processed
        Rectangles.remove(R) # remove this rectangle

        zNW, zSE = R[0], R[1]

        # R2 = [[zNW[0], (zSE[1]+zNW[1])/2 ], zSE] -> R2 not used, but helpful to know for context
        
        z_hat = []
        
        # Lexmin(Z1, Z2) --> set objective to Z1 and find optimal objective 
        m.setObjective(z1_obj)
        if count != 1: # when count is 1, there are no constraints to be removed
            while m.getConstrByName("pre_constraint"):
                m.remove(m.getConstrByName("pre_constraint"))
                m.update()
            while m.getConstrByName("rectangle_bounds"):
                m.remove(m.getConstrByName("rectangle_bounds"))
                m.update()
        m.addConstr(quicksum([variable[i]* objectives[1][i] for i in range(num_x)]) <= (zSE[1]+zNW[1])/2, "rectangle_bounds") #z2_UL
        
        # update model and find optimal objective value
        m.update()
        m.optimize()
        opt_var = [np.round(variable[i].x) for i in range(num_x)] # prevent numerical issues which occur without rounding
        optimal_z1 = int(sum(opt_var*(np.array(objectives[0]))))
        z_hat.append(optimal_z1)
        

        # Use Z* from Lexmin(Z1, Z2) --> and set objective to Z2 to find optimal objective 
        m.setObjective(z2_obj)
        m.addConstr(quicksum([variable[i]*objectives[0][i] for i in range(num_x)]) <= z_hat[0], "pre_constraint")
        
        # update model and find optimal objective value
        m.update()
        m.optimize()
        opt_var = [np.round(variable[i].x) for i in range(num_x)] # prevent numerical issues which occur without rounding
        optimal_z2 = int(sum(opt_var*(np.array(objectives[1])))) 
        z_hat.append(optimal_z2)
        
        z_hat = np.round(z_hat)
        
        # if z_hat does not exactly equal all elements of zSE, it is a new NDP
        if not (z_hat == zSE).all(): 
            foundNDPs.append(z_hat)
            to_add = [z_hat, zSE]
            Rectangles.append(to_add)

        # R3 = [zNW, [z_hat[0] - epsilon, (R[0][1] + R[1][1])/2]] -> R3 not used, but helpful to know for context

        z_tilde = []
        
        # Lexmin(Z2, Z1) --> set objective to Z2 and find optimal objective
        m.setObjective(z2_obj)
        while m.getConstrByName("pre_constraint"):
            m.remove(m.getConstrByName("pre_constraint"))
            m.update()
        while m.getConstrByName("rectangle_bounds"):
            m.remove(m.getConstrByName("rectangle_bounds"))
            m.update()
        m.addConstr(quicksum([variable[i]* objectives[0][i] for i in range(num_x)]) <= z_hat[0] - epsilon, "rectangle_bounds") #z1_UL
        
        # update model and find optimal objective value
        m.update()
        m.optimize()
        opt_var = [np.round(variable[i].x) for i in range(num_x)] # prevent numerical issues which occur without rounding
        optimal_z1 = int(sum(opt_var*(np.array(objectives[1]))))
        z_tilde.append(optimal_z1)

        
        # Use Z* from Lexmin(Z2, Z1) --> and set objective to Z1 to find optimal objective 
        m.setObjective(z1_obj)
        m.addConstr(quicksum([variable[i]*objectives[1][i] for i in range(num_x)]) <= z_tilde[0], "pre_constraint")
        # update model and find optimal objective value
        m.update()
        m.optimize()
        opt_var = [np.round(variable[i].x) for i in range(num_x)] # prevent numerical issues which occur without rounding
        optimal_z2 = int(sum(opt_var*(np.array(objectives[0]))))
        z_tilde.append(optimal_z2)
        
        # reverse order of z_tilde, because z_tilde is currently [z_2, z_1]
        z_tilde = list(reversed(z_tilde))
        z_tilde = np.round(z_tilde)
        
        # if z_hat does not exactly equal all elements of zNW, it is a new NDP
        if not (z_tilde == zNW).all():
            foundNDPs.append(z_tilde)
            to_add = [zNW, z_tilde]
            Rectangles.append(to_add)
    
    # ------------ Return Output ------------
    
    elapsed = (time.time() - start)
    summary = [elapsed, len(foundNDPs), count]
    
    return foundNDPs, summary


def supernal(num_x, constraint_rhs, objectives, constraints):
    # start timer
    start = time.time()
    
    n = num_x
    b = constraint_rhs
    o = objectives
    c = constraints
    
    # ------------ Helper Methods ------------
    
    def FindSupernal(n, b, o, c): # helper method for finding the initial supernal point 
        # use gurobi to solve a multi objective problem
        # set the first objective as 1st priority and decrease the order from there
        m = Model('Find Supernal Point')
        m.setParam("Threads", 1)
        m.Params.OutputFlag = 0
        variable = m.addVars(n, vtype=GRB.BINARY, name="variable")
        obj = LinExpr(quicksum([variable[j]*o[0][j] for j in range(n)]))
    
        m.setObjective(obj, GRB.MAXIMIZE)
        for i in range(1, len(o)):
            obj = LinExpr(quicksum([variable[j]*o[i][j] for j in range(n)]))
            m.setObjectiveN(obj, index=i)
        for k in range(len(c)):
            constraint = m.addConstr(quicksum([c[k][j]*variable[j] for j in range(n)]) <= b[k])
    
        m.update()
        m.optimize()
        variable_Optimal = [np.round(variable[i].x) for i in range(n)] 
        Objective_function_value = m.objVal
    
        #get the Z values for the optimal point that gurobi returned 
        #since this was done using multi objective programming in gurobi we can just ask gurobi to return them
        obj_Val_list = []
        for o1 in range(len(o)):
            m.params.ObjNumber = o1
            obj_Val_list.append(m.ObjNVal)
    
        return variable_Optimal, obj_Val_list


    def RemoveDominatedRegions(Regions):
    #remove dominated regions
        Regions_to_remove2 = []
        for i in Regions:
            for k in Regions:
                #loop through all regions twice, check that you arent comparing the same region
                if i !=k :
                    dominated = True
                    #look to see if the region is dominated, dominated means that all Zi's are smaller than the Zj's
                    for j in range(len(k)):
                        if k[j] > i[j]:
                            dominated = False
                    if dominated == True and k not in Regions_to_remove2:
                        Regions_to_remove2.append(k)
        #remove dominated regions and return the new Regions set
        for reg in Regions_to_remove2:
            Regions.remove(reg)
        return Regions
    
    # ------------ Initializing Parameters and Data Structures ------------
    
    # count number of regions explored
    count = 0
    
    #initialize the returns: foundNDPs will be the x values of the points, FoundNDPs_val will be the z values 
    FoundNDPs = [] 
    FoundNDPs_val = []
    
    #find the supernal point using gurobis multi objective programming
    variable_Optimal, obj_Val_list = FindSupernal(n, b, o, c)
     
    #set the weights for lexmin
    weights = [(1/3), 0.5, 0.2] #start by manually inputting these, can be changed later
    
    # ------------ Initializing Regions ------------
    
    #initialize Regions with the Supernal point
    Regions = [obj_Val_list]
    
    #choose region as the Supernal point 
    region = Regions[0]
    
    #solve weighted sum
    m = Model('Solve weighted sum')
    m.setParam("Threads", 1)
    m.Params.OutputFlag = 0
    #had to reduce the GAP tolerance to remove numerical rounding errors 
    m.Params.MIPGap = 1e-6
    m.setParam(GRB.Param.TimeLimit, 3600)
    
    #create n variables 
    variable = m.addVars(n, vtype=GRB.BINARY, name="variable")
    
    #for each objective, create a linear expression sum(Oi*Xi)
    objs = []    
    for i in range(len(o)):
        obj= LinExpr(quicksum([variable[j]*o[i][j] for j in range(n)]))
        objs.append(obj)
    
    #create a new linear expression sum(weights[i]*objective[i]) using the objectives created above
    obj_new = LinExpr(quicksum([weights[u]*objs[u] for u in range(len(objs))]))
    m.setObjective(obj_new, GRB.MINIMIZE)
    
    #add knapsack constraints, as many as there are
    for k in range(len(c)):
        m.addConstr(quicksum([c[k][j]*variable[j] for j in range(n)]), GRB.LESS_EQUAL, b[k], "knapsackconstraint{0}".format(k))
    
    #add the region constraints (this is just to initialize it, it is updated dynamically as each region is chosen) z value must be less than or equal to the Z value of the region, for each objective
    for i in range(len(o)):
        m.addConstr(quicksum([variable[j]* o[i][j] for j in range(n)]), GRB.LESS_EQUAL, region[i], "constraint{0}".format(i))
        
    m.update()
   
    # ------------ Explore Regions ------------
    
    while len(Regions) > 0:
        #choose a region
        region = Regions[0]
        count +=1
        #region  = Regions[np.random.randint(0, len(Regions))]
        
        #dynamically update the rhs of the region constraint to the chosen region 
        for h in range(len(o)):
            new_rhs = region[h]
            constr = m.getConstrByName("constraint{0}".format(h))
            constr.rhs = new_rhs
            m.update()
        
        #optimize the model
        m.update()
        m.optimize()
        
        #if the model is feasible, ie if there are still points in the region 
        if m.status == 2:
            variable_Optimal = [np.round(variable[i].x) for i in range(n)] 
            
            Objective_function_value = m.objVal
            
            FoundNDPs.append(variable_Optimal)
            
            #calculate the Z value for the optimal point and add this to the FoundNDPs
            z_val_set = []
            for z_val in range(len(o)):
                p = quicksum([variable_Optimal[k]*o[z_val][k] for k in range (n)])
                z_val_set.append(p.getValue())
            
            FoundNDPs_val.append(z_val_set)
            
            Regions_to_add = [] 
            Regions_to_remove = []
            
            for i in Regions:
                #look through each region and check if the Z* is in that region
                in_region = True
                for j in range(len(i)):
                    if z_val_set[j] > i[j]:
                        in_region = False
                if in_region == True:
                    Regions_to_remove.append(i)
                    #if the point is in the region, remove that region
                    #create j new regions based on z*
                    for obj in range(len(o)):
                        new_val = z_val_set[obj]-1
                        new_z = i.copy()
                        new_z[obj] = new_val
                        Regions_to_add.append(new_z)
            
            #outside of the loop go through and add all new regions and remove the regions that should be removed
            for reg in Regions_to_add:
                Regions.append(reg)
            for reg2 in Regions_to_remove:
                Regions.remove(reg2)
            
            #if j >=3, remove dominated regions
            if len(o) >=3:
                Regions = RemoveDominatedRegions(Regions)
        else:
            #if the model is not feasible (ie there are no points in the region), just remove the region 
            Regions.remove(region)
            
    # ------------ Return Output ------------
    
    elapsed = (time.time() - start)
    summary = [elapsed, len(FoundNDPs_val), count]
    
    return FoundNDPs_val, summary


def competition(num_x, constraint_rhs, objectives, constraints): # Supernal With Improvements 
    # start timer
    start = time.time()
    
    n = num_x
    b = constraint_rhs
    o = objectives
    c = constraints
    
    def FindSupernal(n, b, o, c):
        #find the initial supernal point 
        #use gurobi to solve a multi objective problem
        #set the first objective as 1st priority and decrease the order from there
        m = Model('Find Supernal Point')
        m.setParam("Threads", 1)
        m.Params.OutputFlag = 0
        variable = m.addVars(n, vtype=GRB.BINARY, name="variable")
        obj = LinExpr(quicksum([variable[j]*o[0][j] for j in range(n)]))
    
        m.setObjective(obj, GRB.MAXIMIZE)
        for i in range(1, len(o)):
            obj = LinExpr(quicksum([variable[j]*o[i][j] for j in range(n)]))
            m.setObjectiveN(obj, index=i)
        for k in range(len(c)):
            constraint = m.addConstr(quicksum([c[k][j]*variable[j] for j in range(n)]) <= b[k])
    
        m.update()
        m.optimize()
        variable_Optimal = [np.round(variable[i].x) for i in range(n)] 
        Objective_function_value = m.objVal
    
        #get the Z values for the optimal point that gurobi returned 
        #since this was done using multi objective programming in gurobi we can just ask gurobi to return them
        obj_Val_list = []
        for o1 in range(len(o)):
            m.params.ObjNumber = o1
            obj_Val_list.append(m.ObjNVal)
    
        return variable_Optimal, obj_Val_list


    def RemoveDominatedRegions(Regions):
    #remove dominated regions
        Regions_to_remove2 = []
        for i in Regions:
            for k in Regions:
                #loop through all regions twice, check that you arent comparing the same region
                if i !=k :
                    dominated = True
                    #look to see if the region is dominated, dominated means that all Zi's are smaller than the Zj's
                    for j in range(len(k)):
                        if k[j] > i[j]:
                            dominated = False
                    if dominated == True and k not in Regions_to_remove2:
                        Regions_to_remove2.append(k)
                        break
        #remove dominated regions and return the new Regions set
        for reg in Regions_to_remove2:
            Regions.remove(reg)
        return Regions
    
    
    # count number of regions explored
    count = 0
    
    #initialize the returns: foundNDPs will be the x values of the points, FoundNDPs_val will be the z values 
    FoundNDPs = [] 
    FoundNDPs_val = []
    
    
    #find the supernal point using gurobis multi objective programming
    variable_Optimal, obj_Val_list = FindSupernal(n, b, o, c)
     
    #set the weights for lexmin
    weights = [(1/3), 0.5, 0.2, 0.5, 0.666] #start by manually inputting these, can be changed later
    
    
    #initialize Regions with the Supernal point
    Regions = [obj_Val_list]
    
    #choose region as the Supernal point 
    region = Regions[0]
    
    #solve weighted sum
    m = Model('Solve weighted sum')
    m.setParam("Threads", 1)
    m.Params.OutputFlag = 0
    #had to reduce the GAP tolerance to remove numerical rounding errors 
    m.Params.MIPGap = 1e-6
    m.setParam(GRB.Param.TimeLimit, 3600)
    
    #create n variables 
    variable = m.addVars(n, vtype=GRB.BINARY, name="variable")
    
    #for each objective, create a linear expression sum(Oi*Xi)
    objs = []    
    for i in range(len(o)):
        obj= LinExpr(quicksum([variable[j]*o[i][j] for j in range(n)]))
        objs.append(obj)
    
    #create a new linear expression sum(weights[i]*objective[i]) using the objectives created above
    obj_new = LinExpr(quicksum([weights[u]*objs[u] for u in range(len(objs))]))
    m.setObjective(obj_new, GRB.MINIMIZE)
    
    #add knapsack constraints, as many as there are
    for k in range(len(c)):
        m.addConstr(quicksum([c[k][j]*variable[j] for j in range(n)]), GRB.LESS_EQUAL, b[k], "knapsackconstraint{0}".format(k))
    
    #add the region constraints (this is just to initialize it, it is updated dynamically as each region is chosen) z value must be less than or equal to the Z value of the region, for each objective
    for i in range(len(o)):
        m.addConstr(quicksum([variable[j]* o[i][j] for j in range(n)]), GRB.LESS_EQUAL, region[i], "constraint{0}".format(i))
        
    m.update()
   
    
    while len(Regions) > 0:
        #choose a region
        region = Regions[-1]
        count +=1
        #region  = Regions[np.random.randint(0, len(Regions))]
        
        #dynamically update the rhs of the region constraint to the chosen region 
        for h in range(len(o)):
            new_rhs = region[h]
            constr = m.getConstrByName("constraint{0}".format(h))
            constr.rhs = new_rhs
            m.update()
        
        #optimize the model
        m.update()
        m.optimize()
        
        #if the model is feasible, ie if there are still points in the region 
        if m.status == 2:
            variable_Optimal = [np.round(variable[i].x) for i in range(n)] 
            
            
            Objective_function_value = m.objVal
            
            FoundNDPs.append(variable_Optimal)
            
            #calculate the Z value for the optimal point and add this to the FoundNDPs
            z_val_set = []
            for z_val in range(len(o)):
                p = quicksum([variable_Optimal[k]*o[z_val][k] for k in range (n)])
                z_val_set.append(p.getValue())
    
            
            FoundNDPs_val.append(z_val_set)
            
            
            Regions_to_add = [] 
            Regions_to_remove = []
            
            
            for i in Regions:
                #look through each region and check if the Z* is in that region
                in_region = True
                for j in range(len(i)):
                    if z_val_set[j] > i[j]:
                        in_region = False
                if in_region == True:
                    Regions_to_remove.append(i)
                    #if the point is in the region, remove that region
                    #create j new regions based on z*
                    for obj in range(len(o)):
                        new_val = z_val_set[obj]-1
                        new_z = i.copy()
                        new_z[obj] = new_val
                        Regions_to_add.append(new_z)
            
            #outside of the loop go through and add all new regions and remove the regions that should be removed
            for reg in Regions_to_add:
                Regions.append(reg)
            for reg2 in Regions_to_remove:
                Regions.remove(reg2)
            
            #if j >=3, remove dominated regions
            if len(o) >=3:
                Regions = RemoveDominatedRegions(Regions)
        else:
            #if the model is not feasible (ie there are no points in the region), just remove the region 
            Regions.remove(region)

    elapsed = (time.time() - start)
    summary = [elapsed, len(FoundNDPs_val), count]
    
    return FoundNDPs_val, summary


def SolveKnapsack_11(Input_file_directory, method):
    
    num_x, constraint_rhs, objectives, constraints = read_input(Input_file_directory)
    
    if method == 1:
        NDPs, summary = brute_force(num_x, constraint_rhs, objectives, constraints)
        export_NDPs("BF_NDP_11.txt", NDPs)
        export_summary("BF_SUMMARY_11.txt", summary)
    elif method == 2:
        NDPs, summary = rectangular(num_x, constraints, constraint_rhs, objectives)
        export_NDPs("BB_NDP_11.txt", NDPs)
        export_summary("BB_SUMMARY_11.txt", summary)
    elif method == 3:
        NDPs, summary = supernal(num_x, constraint_rhs, objectives, constraints)
        export_NDPs("SP_NDP_11.txt", NDPs)
        export_summary("SP_SUMMARY_11.txt", summary)
    elif method == 4:
        NDPs, summary = competition(num_x, constraint_rhs, objectives, constraints)
        export_NDPs("COMPETITION_2D_NDP_11.txt", NDPs)
        export_summary("COMPETITION_2D_SUMMARY_11.txt", summary)
    elif method == 5:
        NDPs, summary = competition(num_x, constraint_rhs, objectives, constraints)
        export_NDPs("COMPETITION_3D_NDP_11.txt", NDPs)
        export_summary("COMPETITION_3D_SUMMARY_11.txt", summary)
    return

