# Define your reaction components
reactants1 = {
    '2-{5-Chloro-3-[(R)-3-morpholinyl]phenyl}-4,4,5,5-tetramethyl-1,3,2-dioxaborolane': 'IC(C=C1Cl)=C(C)C(N)=N1',
    'tert-Butyl (R)-3-[5-chloro-3-(4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl)phenyl]-4-morpholinecarboxylate': 'CC(C)(C)OC(=O)N1CCOC[C@H]1c3cc(Cl)cc(B2OC(C)(C)C(C)(C)O2)c3'
}

reactants2 = {
    '2-amino-6-chloropyrazine': 'Nc1ccnc(Cl)n1',
    '4-[tert-Butoxycarbonyl-tert-butyl(oxycarbonylamino)]-2-chloropyrimidine': 'CC(C)(C)OC(=O)N(C(=O)OC(C)(C)C)c1ccnc(Cl)n1'
}

catalysts = {
    'Pd(dppf)Cl2': '[Fe].Cl[Pd]Cl.[CH]1[CH][CH][C]([CH]1)P(c2ccccc2)c3ccccc3.[CH]4[CH][CH][C]([CH]4)P(c5ccccc5)c6ccccc6',
    'Pd[P(Ph)3]2Cl2': 'P(C1C=CC=CC=1)(C1=CC=CC=C1)C1C=CC=CC=1.P(C1C=CC=CC=1)(C1C=CC=CC=1)C1C=CC=CC=1.[Pd](Cl)Cl',
    'Pd(OAc)2': 'CC(=O)O[Pd]OC(C)=O'
}

solvents = {
    'Toluene': 'CC(=CC=C1)C=C1',
    'Water': 'O',
    'MeCN': 'CC#N',
    'DMAc': 'CN(C)C(C)=O',
    '2-MeTHF': 'CC1CCCO1',
    'iPAc': 'CC(C)OC(C)=O'
}

bases = {
    'K3PO4': '[O-]P(=O)([O-])[O-].[K+].[K+].[K+]',
    'K2CO3': 'O=C(O[K])O[K]',
    'Cs2CO3': 'O=C(O[Cs])O[Cs]',
    'Na2CO3': 'O=C(O[Na])O[Na]',
    'DBU': 'C2CCC1=NCCCN1CC2'
}