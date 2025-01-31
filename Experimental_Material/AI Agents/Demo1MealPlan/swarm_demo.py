#Tooling for running reliable multi-AI-agent orchestration
#Test use Case: Meal-planner app

#Tools and Corresponding Routines:
# note_preference: Write down user meal preferences, dietary restrictions
# note_meal: Write down agreed-upon meals
# save_recipe: Create recipe, record
# add_to_grocery_list: Add necessary grocieries for each recipe to list
# generate_final_report: Collate all information

#Three Routines
# Triage Agent - Processing initial request, report gen
# Meal Planner - Note preference, note meal, return to triage
# Recipe and Grocery Agent - Make recipe, add to groceries, return to triage

from swarm import Swarm, Agent
from swarm.repl import run_demo_loop
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your actual key

# Initialize Swarm
client = Swarm()

context_variables = {
    "preferences": [],
    "meals": [],
    "recipes": [],
    "grocery_list": []
}

def note_preference(preference: str):
    """Make a note of dietary preference to keep track of"""

    #Append to file
    with open("/Users/supriyarai/Code/personalLLM/Experimental_Material/AI Agents/Demo1MealPlan/preferences.txt", "a") as file:
        file.write(f"{preference}\n")

    context_variables['preferences'].append(preference)

    return "Successfully noted preference!"

def note_meal(meal: str):
    """Write down meal agreed upon"""

    #Append to file
    with open("/Users/supriyarai/Code/personalLLM/Experimental_Material/AI Agents/Demo1MealPlan/meals.txt", "a") as file:
        file.write(f"{meal}\n")

    context_variables['meals'].append(meal)

    return "Successfully noted meal!"

def save_recipe(recipe: str):
    """Create recipe, record"""

    #Append to file
    with open("/Users/supriyarai/Code/personalLLM/Experimental_Material/AI Agents/Demo1MealPlan/recipes.txt", "a") as file:
        file.write(f"{recipe}\n")

    context_variables['preferences'].append(recipe)

    return "Successfully recorded recipe!"

def add_to_grocery_list(items: str):
    """Make a note of dietary preference to keep track of"""

    #Append to file
    with open("/Users/supriyarai/Code/personalLLM/Experimental_Material/AI Agents/Demo1MealPlan/grocery_list.txt", "a") as file:
        file.write(f"{items}\n")

    context_variables['preferences'].append(items)

    return "Successfully noted preference!"

def generate_final_report(context_variables: dict):
    """Collate all information from context variables"""
    report = "Meal Planning Report\n===============\n\n"

    #Add preferences
    report += "Dietary Preferences and Restrictions:\n"
    for pref in context_variables.get('preferences', []):
        report += f"- {pref}\n"
    report += "\n"

    #Add meals
    report += "Planned Meals:\n"
    for meal in context_variables.get('meals', []):
        report += f"- {meal}\n"
    report += "\n"

    # Add recipes
    report += "Recipes:\n"
    for recipe in context_variables.get('recipes', []):
        report += f"- {recipe}\n"
    report += "\n"

    # Add grocery list
    report += "Grocery List:\n"
    for item in context_variables.get('grocery_list', []):
        report += f"- {item}\n"

    #Write report to file
    with open("/Users/supriyarai/Code/personalLLM/Experimental_Material/AI Agents/Demo1MealPlan/final_report.txt", "w") as file:
        file.write(report)

    return "Final report generated successfully! Saved as final_report.txt in directory."

def transfer_back_to_triage():
    """Transfer back to triage agent once task is complete, or question out of scope is asked"""
    return triage_agent

def transfer_to_recipe_agent():
    """Transfer to Recipe Agent for hanndling creation of recipes for the meals."""
    return recipe_agent

def transfer_to_meals_agent():
    """Transfer to Meals Agent for handling preferences, dietary restrictions, and meals."""
    return meal_agent

triage_system_message = (
    "You are an expert triaging agent for a meal prepping orchestation, handling the conversation between the user and multiple specialist agents."    
    "Once you are ready to transfer to the right intent, call the tool to transfer to the right agent. Do not handle requests yourself"
    "You dont need to know specifics, just the topic of the request."
    "When you need more information to triage the request to an agent, ask a direct question without explaining why you're asking it."
    "Do not share your thought process with the user! Do not make unreasonable assumptions on behalf of user."
    "Based on the conversation or when its first starting out, suggest some possible actions to take to accurately transfer the user"
    "Finally, you have the ability to compile the conversation into a report. If it looks like the user is at a good stopping point, suggest this."
    )

meal_system_message = (
    "You are a meal planning assistant."
    "Always answer in a sentence or less."
    "Always take explicit note of preferences or restrictions"
    "Follow the following routine with the user:"
    "Find all restrictions and preferences, make explicit note of them"
    "Then, help ideate different meal choices, noting down agreed upon meals"
    "Try and get to diverse 3 different meal choices"
    "Make your questions subtle and natural."
    "Once you have three meals transfer back to the triage agent."
)

recipe_system_message = (
    "You are a recipe creation assistant."
    "Keep your conversation answers brief to maintain flow"
    "Follow the following routine with the user:"
    "Determine what meals the user would like to get recipes for"
    "Then, create recipes for the meals and save them for the user"
    "Make sure to add the ingredients needed to the grocery list"
    "As well as anything else the user might need for groceries"
    "Make your responses subtle and natural."
    "At the end of the conversation, transfer back to the triage agent."
)

# Agents
triage_agent = Agent(
    name="Triage Agent",
    model="gpt-4o",
    instructions=triage_system_message,
    functions=[transfer_to_meals_agent, transfer_to_recipe_agent, generate_final_report]
)

meal_agent = Agent(
    name="Meal Agent",
    model="gpt-4o",
    instructions=meal_system_message,
    functions = [note_meal, note_preference, transfer_back_to_triage]
)

recipe_agent = Agent(
    name="Recipe Agent",
    model="gpt-4o",
    instructions=recipe_system_message,
    functions = [save_recipe, add_to_grocery_list, transfer_back_to_triage]
)

run_demo_loop(triage_agent, stream=True, context_variables=context_variables)