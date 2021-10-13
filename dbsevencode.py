def even(s,n):
    i=2
    c=0
    while(True):
        if i%2==0:
            print(i,end=" ")
            c=c+1
            if c==4:
                break
        i=i+1
s,n=map(int,input().split())
n1=even(s,n)


'''
s=2
n=4
output= 2 4 6 8 '''
