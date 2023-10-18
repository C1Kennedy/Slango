# Cameron Kennedy, Blank Bruno, Bella Quintero, UTEP, Fall 2023
# Speech and Language Processing
# P4: Project Prototype

def generateQuestion(num):
    # Initialize empty set.
    questions = set()
    
    for i in num:
        sentence = ""
        # Select random word from data structure where likelihoods are contained.
        # Append to sentence.
        # Repeat until [TBD].
        # Add sentence to questions.
        questions.add(sentence)
    
    # Return set of questions.
    return questions

if __name__ == "__main__":
    print("Welcome to Slango. Are you ready to test your knowledge of slang?")
    # Open score file.
    # Set score equal to value in score file.
    
    score = 0
    print("Your max score is currently:", 0)
    limit = int(input("How many failed attempts will you allow?"))
    # Get input, set limit equal to input.
    attempts = 0
    print("Enter 'Y' to begin, 'N' to cancel at any time.")
    
    # While 'N' is not entered, run program.
    while ((input() != 'N') and (attempts < limit)):
        print(5)
        attempts += 1
    
    