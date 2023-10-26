# Cameron Kennedy, Blank Bruno, Bella Quintero, UTEP, Fall 2023
# Speech and Language Processing
# P4: Project Prototype

import random

def generateQuestion(question_set):
    question_num = random.randint(0, 2)
    
    print(question_set[question_num][0])
    print("Options:")
    for i in question_set[question_num][1]:
        print(i)
        
    return question_set[question_num][2]
    

if __name__ == "__main__":
    # Intro.
    print("Welcome to Slango. Are you ready to test your knowledge of slang?")
    
    question_set = []
    question_set.append(["He was extremely ___ that he had lost.", ["chuffed", "salty", "chill"], "salty"])
    question_set.append(["After the announcement, they were on the ___.", ["vibe", "crash", "copium"], "copium"])
    question_set.append(["___, she said after the plan was described.", ["bet", "hooked", "hits different"], "bet"])
    
    # Open score file.
    # Set score equal to value in score file.
    file = open('highScore.txt', 'r')
    score = file.read()
    file.close()
    
    # If score exists, convert to int. Else reset to 0.
    if (score != ''):
        score = int(score)
        
    else:
        score = 0
    
    # Display max score.
    # Prompt for max failures allowed.
    # Set current score to 0.
    print("Your max score is currently:", score)
    limit = int(input("How many failed attempts will you allow? "))
    failures, curr_score = 0, 0
    
    # Allow user to start game whenever they are ready by entering 'Y'.
    print("Enter 'Y' to begin!")
    while (input() != 'Y'):
        continue
    
    # While the number of incorrectly-answered questions is less than the
    # limit, continue.
    while (failures < limit):
        
        # Call the generateQuestion function and store correct answer to variable.
        correct_ans = generateQuestion(question_set)
        
        # Check if input is equal to correct answer. Increase score if true, failures if false.
        if (input().casefold() == correct_ans):
            curr_score += 1
        else:
            failures += 1
            
        print()
    
    # Write current score to file if greater than high score.
    if (curr_score > score):
        file = open('highScore.txt', 'w')
        file.write(str(curr_score))
        file.close()
        
    print("Thank you for playing Slango!")
    
    