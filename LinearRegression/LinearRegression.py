# -*- coding: utf-8 -*-3
"""
Created on Mon Jan 30 17:28:51 2017

@author: Rodolfo Braga Martini
"""
import numpy as np
import matplotlib as plt

def compute_error_for_line_given_points(b, m, points):
    #Calculate the difference between every point and the slope and then square it (So it remains always positive)
    #Calculate the mean of the sum of all errors
    #Error(b,m) = 1/N * SUM((y - (mx + b))²)
    #Step 1 - Initialize error
    totalError = 0
    #For every point
    for i in range(0, len(points)):
        #Get X values
        x = points[i, 0]
        #Get Y values
        y = points[i, 1]
        #Calculate difference, square it and add to totalError
        totalError += ((y - (m*x + b))**2)
    #Return the average    
    return totalError / float(len(points))

def gradient_step(current_b, current_m, points, learning_rate):
    #Starting points for the gradient
    #Partial derivative formula with respect to b
        # m = 2/N * -SUM(x * (y-(mx+b)))
        # b = 2/N * -SUM(y - (mx+b))
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    #Computing all partial derivatives wich is a function to find on 3D graphs (x,y,error in this case) the minimum local throughout many interactions
    #wich is the point where the error is the smallest, therefore is the point where we have our optimal b and m
    #The function does not give the optimal at once, but it points the direction through wich the error decreases
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #Partial derivatives formula
        b_gradient += - 2/N * (y - (current_m*x + current_b))
        m_gradient += - 2/N * (x * (y - (current_m*x + current_b)))
    #Now we update b and m using the partial derivatives
    new_b = current_b - (learning_rate * b_gradient)
    new_m = current_m - (learning_rate * m_gradient)
    return [new_b, new_m]    

def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    #Inicitalize b and m so we keep a diferent variable that is not the initial anymore
    b = initial_b
    m = initial_m
    
    #Perform gradient descent
    for i in range(num_iterations):
        #We update b and m at every loop with new more accurate b and m
        b, m = gradient_step(b, m, points, learning_rate)
    return [b, m]

def run():   
    #Step 1 - Collect data (Hours of study x Test score)
    graphic_points = np.genfromtxt('data.csv', delimiter = ',')
    
    #Step 2 - Define Hyperparameters
    #Define speed of convergion (Achievement of the optmum state) - If it's too big the model will not converge
    # SO this must be a balance
    learning_rate = 0.0001
    #y = mx + b --> Used to draw the graph of the slope // Slope formula. 
    # x,y are the independent and dependent variables in this order
    # m is the inclination os the slope (calculated by the Tg of the angle between the slope and the X axis)
    # b is the point where the slope crosses the Y axis
    initial_m = 0
    initial_b = 0
    #Small dataset, so why feel iterations
    num_iterations = 1000
    
    #Step 3 - Train the model
    print("Strating gradient descent at b = {0}, m = {1} and error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, graphic_points)))
    print("Running")
    [b, m] = gradient_descent_runner(graphic_points, initial_b, initial_m, learning_rate, num_iterations)
    
    print("Ending point at b = {1}, m = {2} and error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, graphic_points)))
    

if __name__ == '__main__':
    run()
    


