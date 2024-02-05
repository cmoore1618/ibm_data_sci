import ephem

def find_full_moons(year):
    full_moons = []
    # Start from the beginning of the year
    date = f'{year}-01-01'
    while True:
        # Use the next_full_moon method to find the next full moon
        next_full_moon = ephem.next_full_moon(date)
        next_full_moon_date = next_full_moon.datetime().date()
        
        # Check if the full moon is still in the target year
        if next_full_moon_date.year != year:
            break
        
        full_moons.append(next_full_moon_date)
        # Update the date for the next iteration to the day after the current full moon
        date = next_full_moon_date.strftime('%Y-%m-%d')
        
        # Increment by one day to avoid finding the same full moon again
        date = ephem.Date(date) + 1
        
    return full_moons

# Calculate and print the full moons for 2024
full_moons_2024 = find_full_moons(2024)
for date in full_moons_2024:
    print(date)
