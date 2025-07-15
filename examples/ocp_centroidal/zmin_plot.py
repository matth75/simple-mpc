import subprocess
import matplotlib.pyplot as plt

T_ss = [30 + i for i in range(20)]

results = []
for t in T_ss:
    result = subprocess.run(["python", "examples/ocp_centroidal/go2_centroidal.py", str(t)], 
                        capture_output=True, 
                        text=True)
    results.append(float(result.stdout))

print(results)

plt.plot(T_ss, results, "x-")
plt.grid()
plt.title("Min values for com z during jump with 100 iterations horizon")
plt.xlabel("T_ss")
plt.ylabel("Min z")
plt.show()