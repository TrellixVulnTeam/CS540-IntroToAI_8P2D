from dataclasses import dataclass
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # Question 2
    csv_file = sys.argv[1]
    # csv_file = "hw5.csv"
    # csv_file = "toy.csv"
    headers = ["year", "days"]

    df = pd.read_csv(csv_file)
    ax = df.plot(x = headers[0], y = headers[1])
    ax.legend().set_visible(False)
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.xticks(np.arange(df["year"].min(), df["year"].max() + 1, 15))
    plt.savefig("plot.jpg") # Save the plot to plot.jpg

    # Question 3

    year_list = df['year'].to_list()
    day_list = df['days'].to_list()
    len_ylist = len(year_list)

    X = np.zeros((len_ylist, 2), dtype='int64')
    Y = []
    for i in range(len_ylist):
        X[i][0] = 1
        X[i][1] = year_list[i]
        Y.append(day_list[i])

    # Q3a
    print("Q3a:")
    print(X)

    # Q3b
    Y = np.array(Y)
    print("Q3b:")
    print(Y)

    # Q3c
    print("Q3c:")
    X_trans = np.transpose(X)
    id_matrix = np.dot(X_trans, X)
    print(id_matrix)

    # Q3d
    print("Q3d:")
    inverse = np.linalg.inv(id_matrix)
    print(inverse)

    # Q3e
    print("Q3e:")
    id_dot_trans = np.dot(inverse, X_trans)
    print(id_dot_trans)

    # Q3f
    print("Q3f:")
    beta_hat = np.dot(id_dot_trans, Y)
    print(beta_hat)

    # Q4
    y_hat = beta_hat[0] + np.dot(beta_hat[1], 2021)
    print("Q4: " + str(y_hat))

    # Q5a
    print("Q5a:")
    if(beta_hat[1] > 0):
        print('>')
    elif(beta_hat[1] < 0):
        print('<')
    else:
        print('=')

    # Q5b
    print("Q5b: The meaning of the sign of the regression slope of Mendota ice is that: ")
    print("If the number of frozen days decreases as the year increases, the sign is '<'.")
    print("If the number of frozen days increases as the year increases, the sign is '>'.")
    print("If the number of frozen days is constant as the year increases, the sign is '='.")
    print("Since the sign for the data is '<', it means that the number of frozen days per year is descreasing every year.")

    # Q6a
    print("Q6a:")
    year_x = (0 - beta_hat[0])/beta_hat[1]
    print(year_x)

    # Q6b
    print("Q6b: ")
    print("Since the regression slope is negative in our example, and thus the number of frozen days is decreasing per year,")
    print("the year ", year_x, " seems like a compelling prediction for when the lake will stop freezing over/ the year where ")
    print("there are no more frozen days.")