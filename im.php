<html>
<body>

<?php 
$command = escapeshellcmd('src/bio.py');
$output = shell_exec($command);
echo $output;
exec('whoami');
?>

</body>
</html>