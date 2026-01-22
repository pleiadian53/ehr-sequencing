echo "Backing up key tpheno modules ..."
echo "> Modules to backup: 1. seqmaker 2. batchpheno 3. ref 4. pattern 5. bin 6. demo (x) 7. cluster 8. sample ..."
tar czf tpheno-key.tar.gz *.py batchpheno ref seqmaker pattern bin cluster sample 

echo "PS: Not backing up configuration directoreis and files (e.g. config, set_env.sh)"
