#Main file which will call the retrieval and response generation functions based on the role.
print("Welcome to FinSolve's AI Assistant...")
role = input("Enter your role(hr/eng): ")

if role=='hr':
    from hr_vector_db import hr_generate_response
    hr_generate_response()

elif(role == 'engineer'):
    from engineer_vector_db import engineer_generate_response

    if(role == 'engineer'):
        engineer_generate_response()

elif(role == 'sales'):
    #Will me writing later
    pass
    
    # More roles can be added here