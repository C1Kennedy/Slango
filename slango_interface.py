# Cameron Kennedy, Blank Bruno, Bella Quintero, UTEP, Fall 2023
# Speech and Language Processing
# Project: Slango Interface

from slango_model import main

if __name__ == "__main__":
    # Intro.
    print("Welcome to Slango. Are you ready to test your knowledge of slang?")
    
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
    
    # While the number of incorrectly answered questions is less than the
    # max number of failures allowed, do the following.
    while (failures < limit):
        print("Building question...")
        
        # Call the slango_model.main() function and store correct answer 
        # as well as answer index to variable.
        index, answer = main()
        
        # Get input.
        l = input()
        
        # If input matches word disregarding case, increment current score.
        if (l.lower() == answer):
            print("Correct!")
            curr_score += 1

            
        else:
            try:
                # If the player inputs the word index, increment current score.
                if (int(l) == (index + 1)):
                    print("Correct!")
                    curr_score += 1
                else:
                    print(f'Incorrect. The actual word for this definition is {answer}.')
                    failures += 1
                    
            except:
                # If neither of the correct answer conditions apply, increment failures.
                print(f'Incorrect. The actual word for this definition is {answer}.')
                failures += 1
            
        print()
    
    # Write current score to file if greater than high score.
    if (curr_score > score):
        print(f'Congratulations! You have reached a new high score of {curr_score}!')
        file = open('highScore.txt', 'w')
        file.write(str(curr_score))
        file.close()
        
    print("Thank you for playing Slango!")
    
    