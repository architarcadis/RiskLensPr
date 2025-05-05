import cairosvg

# Convert Arcadis logo
cairosvg.svg2png(url='./assets/arcadis_logo.svg', write_to='./assets/arcadis_logo.png', output_width=400, output_height=100)

print("Conversion complete")