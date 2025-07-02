#Main file which will call the retrieval and response generation functions based on the role.
from hr_vector_db import HR_BREAK_PROGRAM
from engineer_vector_db import ENGINEER_BREAK_PROGRAM
from finance_vector_db import FINANCE_BREAK_PROGRAM
print("Welcome to FinSolve's AI Assistant...\nSelect your role to proceed \na-HR\nb-Engineer): ")

role = input("Enter your role(a or b): ")

while HR_BREAK_PROGRAM or ENGINEER_BREAK_PROGRAM or FINANCE_BREAK_PROGRAM != True:
    if role.lower()=='a':
        from hr_vector_db import hr_generate_response
        hr_generate_response()

    elif(role.lower() == 'b'):
        from engineer_vector_db import engineer_generate_response
        engineer_generate_response()

    else:
        print("Invalid role selected. Please choose either 'a' for HR or 'b' for Engineer.")
        
        # More roles can be added here 